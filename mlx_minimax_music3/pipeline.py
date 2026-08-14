from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from .condition_encoder import ConditionEncoder
from .config import GenerationConfig, ModelConfig, OptimizationConfig
from .flow_transformer import FlowTransformer
from .language_model import LanguageModel
from .prompt import build_cfg_token_ids
from .rvq_decoder import RVQDepthDecoder, embed_audio_frame, generate_depth_codes
from .sampling import sample_top_k, semantic_guided_logits
from .scheduler import (
    blend_overlap,
    carry_window,
    chunk_starts,
    euler_step,
    flow_timesteps,
    restore_overlap,
    waveform_crop,
)
from .vocoder import Vocoder


class TokenizerLike(Protocol):
    def encode(self, text: str) -> Any: ...


DIT_COMPILE_MAX_LENGTH = 384
DIT_COMPILE_MIN_STEPS = 12


@dataclass
class PipelineComponents:
    tokenizer: TokenizerLike
    language_model: LanguageModel
    rvq_depth_decoder: RVQDepthDecoder
    condition_encoder: ConditionEncoder
    transformer: FlowTransformer
    vocoder: Vocoder


@dataclass(frozen=True)
class GenerationResult:
    audio: mx.array
    sampling_rate: int
    num_frames: int
    num_chunks: int


def _token_ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    if not isinstance(ids, (list, tuple, np.ndarray)):
        raise TypeError("tokenizer.encode must return ids or an object with an ids attribute")
    return [int(token) for token in ids]


def _replace_prefix(sequence: mx.array, prefix: mx.array, length: int) -> mx.array:
    if length == 0:
        return sequence
    return mx.concatenate((prefix[:, :length], sequence[:, length:]), axis=1)


def _flow_cfg_velocity(
    transformer: FlowTransformer,
    latents: mx.array,
    timestep: mx.array,
    condition: mx.array,
    cfg_scale: float,
    *,
    batched: bool,
    prepared_cfg_condition: mx.array | None = None,
) -> mx.array:
    if prepared_cfg_condition is None:
        unconditional_condition = mx.zeros_like(condition)
        prepared_cfg_condition = (
            mx.concatenate((condition, unconditional_condition), axis=0)
            if batched
            else unconditional_condition
        )
    if not batched:
        conditional = transformer(latents, timestep, condition)
        unconditional = transformer(latents, timestep, prepared_cfg_condition)
    else:
        predictions = transformer(
            mx.concatenate((latents, latents), axis=0),
            mx.concatenate((timestep, timestep), axis=0),
            prepared_cfg_condition,
        )
        conditional, unconditional = mx.split(predictions, 2, axis=0)
    return unconditional + cfg_scale * (conditional - unconditional)


class MiniMaxMusic3Pipeline:
    def __init__(
        self,
        components: PipelineComponents,
        model_config: ModelConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
    ):
        self.components = components
        self.model_config = model_config or ModelConfig()
        self.optimization_config = optimization_config or OptimizationConfig.from_env()
        self._compiled_dit = None
        if self.optimization_config.pruned_semantic_head:
            self.components.language_model.prepare_semantic_head(
                self.model_config.audio_code_offset,
                self.model_config.semantic_vocab_size,
                self.model_config.audio_end_token_id,
            )
        self._validate_components()

    def _flow_transformer(self, sequence_length: int, inference_steps: int):
        should_compile = (
            self.optimization_config.compiled_dit
            and sequence_length <= DIT_COMPILE_MAX_LENGTH
            and inference_steps >= DIT_COMPILE_MIN_STEPS
        )
        if not should_compile:
            return self.components.transformer
        if self._compiled_dit is None:
            self._compiled_dit = mx.compile(self.components.transformer)
        return self._compiled_dit

    def _validate_components(self) -> None:
        config = self.model_config
        language = self.components.language_model.config
        decoder = self.components.rvq_depth_decoder.config
        condition = self.components.condition_encoder.config
        transformer = self.components.transformer.config
        vocoder = self.components.vocoder.config

        if decoder.num_codebooks != config.num_codebooks:
            raise ValueError("RVQ num_codebooks does not match model config")
        if decoder.audio_vocab_size != config.audio_vocab_size:
            raise ValueError("RVQ audio_vocab_size does not match model config")
        if decoder.hidden_size != language.hidden_size:
            raise ValueError("language and RVQ hidden sizes must match")
        if condition.num_condition_layers != config.num_codebooks:
            raise ValueError("condition layers must equal num_codebooks")
        if condition.condition_hidden_dim != language.hidden_size:
            raise ValueError("condition and language hidden sizes must match")
        if condition.out_dim != transformer.condition_dim:
            raise ValueError("condition output and transformer condition dimensions must match")
        if transformer.in_channels != config.latent_channels:
            raise ValueError("transformer latent channels do not match model config")
        if vocoder.latent_channels != config.latent_channels:
            raise ValueError("vocoder latent channels do not match model config")
        if vocoder.sampling_rate != config.sampling_rate:
            raise ValueError("vocoder sampling rate does not match model config")
        highest_token = max(
            config.audio_end_token_id,
            config.audio_cfg_token_id,
            config.audio_code_offset + config.semantic_vocab_size - 1,
        )
        if highest_token >= language.vocab_size:
            raise ValueError("language vocabulary does not contain Music3 audio tokens")

    def encode_prompt(self, prompt: str, lyrics: str) -> mx.array:
        ids = build_cfg_token_ids(
            prompt,
            lyrics,
            lambda text: _token_ids(self.components.tokenizer, text),
            self.model_config,
        )
        return mx.array(ids, dtype=mx.int32)

    def generate_frame_hiddens(
        self,
        text_ids: mx.array,
        generation: GenerationConfig,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        generation.validate()
        max_frames = generation.max_frames(self.model_config)
        if text_ids.ndim != 2 or text_ids.shape[0] != 2:
            raise ValueError("text_ids must have shape [2, sequence_length]")

        language = self.components.language_model
        decoder = self.components.rvq_depth_decoder
        cache = [KVCache() for _ in language.layers]
        hidden = language.hidden_states(text_ids, cache=cache)
        last_hidden = hidden[:, -1]

        compact_layout = language.semantic_head_layout
        if compact_layout is None:
            vocab = mx.arange(language.config.vocab_size, dtype=mx.int32)
            start = self.model_config.audio_code_offset
            stop = start + self.model_config.semantic_vocab_size
            allowed_vocab = ((vocab >= start) & (vocab < stop)) | (
                vocab == self.model_config.audio_end_token_id
            )
        else:
            expected_layout = (
                self.model_config.audio_code_offset,
                self.model_config.semantic_vocab_size,
                self.model_config.audio_end_token_id,
            )
            if compact_layout != expected_layout:
                raise ValueError("semantic head layout does not match model config")
            allowed_vocab = mx.ones(
                (self.model_config.semantic_vocab_size + 1,),
                dtype=mx.bool_,
            )

        frame_hiddens: list[mx.array] = []
        for frame_index in range(max_frames + 1):
            logits = language.logits(last_hidden)
            guided = semantic_guided_logits(
                logits,
                allowed_vocab,
                cfg_scale=generation.ar_cfg_scale,
                conditional_top_k=generation.top_k,
            )
            sampled, key = sample_top_k(guided, key, top_k=generation.top_k)
            mx.eval(sampled)
            sampled_id = sampled.item()
            if compact_layout is None:
                if sampled_id == self.model_config.audio_end_token_id:
                    break
                semantic_code = (
                    sampled - self.model_config.audio_code_offset
                ).astype(mx.int32)
            else:
                if sampled_id == self.model_config.semantic_vocab_size:
                    break
                semantic_code = sampled.astype(mx.int32)
            semantic_pair = mx.repeat(semantic_code, 2, axis=0)
            frame_codes, depth_hidden, key = generate_depth_codes(
                language,
                decoder,
                last_hidden,
                semantic_pair,
                key,
                cfg_scale=generation.ar_cfg_scale,
                top_k=generation.top_k,
                model_config=self.model_config,
                use_cache=self.optimization_config.depth_kv_cache,
            )
            if frame_index > 0:
                frame_hiddens.append(mx.concatenate((last_hidden[:1], depth_hidden), axis=-1))
                if len(frame_hiddens) >= max_frames:
                    break

            feedback = embed_audio_frame(
                language,
                decoder,
                frame_codes,
                model_config=self.model_config,
            )
            hidden = language.hidden_states(input_embeddings=feedback, cache=cache)
            last_hidden = hidden[:, -1]
            mx.eval(last_hidden)

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames")
        output = mx.stack(frame_hiddens, axis=1)
        mx.eval(output)
        return output, key

    def denoise_chunks(
        self,
        frame_hiddens: mx.array,
        generation: GenerationConfig,
        key: mx.array,
    ) -> tuple[list[mx.array], mx.array]:
        generation.validate()
        starts = chunk_starts(frame_hiddens.shape[1])
        latent_chunks: list[mx.array] = []
        previous_latent: mx.array | None = None
        previous_condition: mx.array | None = None
        timesteps = flow_timesteps(generation.num_inference_steps)

        for start in starts:
            end = min(start + 200, frame_hiddens.shape[1])
            condition = self.components.condition_encoder(frame_hiddens[:, start:end])
            overlap = 0
            if previous_latent is not None and previous_condition is not None:
                overlap = min(previous_latent.shape[1], condition.shape[1])
                condition = _replace_prefix(condition, previous_condition, overlap)
            transformer = self._flow_transformer(
                condition.shape[1],
                generation.num_inference_steps,
            )
            unconditional_condition = mx.zeros_like(condition)
            prepared_cfg_condition = (
                mx.concatenate((condition, unconditional_condition), axis=0)
                if self.optimization_config.batched_dit_cfg
                else unconditional_condition
            )
            mx.eval(prepared_cfg_condition)

            key, noise_key = mx.random.split(key, 2)
            latents = mx.random.normal(
                (1, condition.shape[1], self.model_config.latent_channels),
                dtype=condition.dtype,
                key=noise_key,
            )
            noise_prompt = latents[:, :overlap]
            for timestep in timesteps:
                if overlap:
                    latents = blend_overlap(
                        latents,
                        noise_prompt,
                        previous_latent,
                        overlap,
                        timestep,
                    )
                time_batch = mx.broadcast_to(timestep, (latents.shape[0],))
                velocity = _flow_cfg_velocity(
                    transformer,
                    latents,
                    time_batch,
                    condition,
                    generation.flow_cfg_scale,
                    batched=self.optimization_config.batched_dit_cfg,
                    prepared_cfg_condition=prepared_cfg_condition,
                )
                latents = euler_step(latents, velocity, generation.num_inference_steps)
                mx.eval(latents)

            if overlap:
                latents = restore_overlap(latents, previous_latent, overlap)
            previous_latent = carry_window(latents)
            previous_condition = carry_window(condition)
            latent_chunks.append(latents)
            mx.eval(previous_latent, previous_condition, latents)

        return latent_chunks, key

    def decode_chunks(self, latent_chunks: list[mx.array]) -> mx.array:
        if not latent_chunks:
            raise ValueError("latent_chunks must not be empty")
        waveforms = []
        for index, latents in enumerate(latent_chunks):
            waveform = self.components.vocoder(latents)
            crop = waveform_crop(
                index,
                len(latent_chunks),
                self.components.vocoder.config.hop_length,
            )
            end = waveform.shape[-1] - crop.right if crop.right else waveform.shape[-1]
            waveforms.append(waveform[..., crop.left:end])
        audio = mx.clip(mx.concatenate(waveforms, axis=-1).astype(mx.float32), -1.0, 1.0)
        mx.eval(audio)
        return audio

    def generate(
        self,
        prompt: str,
        lyrics: str,
        generation: GenerationConfig | None = None,
    ) -> GenerationResult:
        generation = generation or GenerationConfig()
        key = mx.random.key(generation.seed)
        text_ids = self.encode_prompt(prompt, lyrics)
        frame_hiddens, key = self.generate_frame_hiddens(text_ids, generation, key)
        latent_chunks, _ = self.denoise_chunks(frame_hiddens, generation, key)
        audio = self.decode_chunks(latent_chunks)
        return GenerationResult(
            audio=audio,
            sampling_rate=self.model_config.sampling_rate,
            num_frames=frame_hiddens.shape[1],
            num_chunks=len(latent_chunks),
        )
