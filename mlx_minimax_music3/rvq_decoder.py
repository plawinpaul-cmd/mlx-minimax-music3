from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language_model import LanguageModel
from .sampling import sample_top_k


@dataclass(frozen=True)
class RVQDecoderConfig:
    hidden_size: int = 4096
    num_layers: int = 4
    num_attention_heads: int = 16
    intermediate_size: int = 6144
    audio_vocab_size: int = 1024
    num_codebooks: int = 8
    max_position_embeddings: int = 16
    rms_norm_eps: float = 1e-6


class DepthAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, length, hidden_size = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.to_q(hidden_states).reshape(shape).transpose(0, 2, 1, 3)
        key = self.to_k(hidden_states).reshape(shape).transpose(0, 2, 1, 3)
        value = self.to_v(hidden_states).reshape(shape).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask="causal",
        )
        return self.to_out(output.transpose(0, 2, 1, 3).reshape(batch, length, hidden_size))


class RVQDecoderBlock(nn.Module):
    def __init__(self, config: RVQDecoderConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DepthAttention(config.hidden_size, config.num_attention_heads)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        normalized = self.post_attention_layernorm(hidden_states)
        gated = nn.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return hidden_states + self.down_proj(gated)


class RVQDepthDecoder(nn.Module):
    def __init__(self, config: RVQDecoderConfig | None = None):
        super().__init__()
        self.config = config or RVQDecoderConfig()
        residual_vocab = self.config.audio_vocab_size * (self.config.num_codebooks - 1)
        self.audio_embeddings = nn.Embedding(residual_vocab, self.config.hidden_size)
        self.projection = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )
        self.layers = [RVQDecoderBlock(self.config) for _ in range(self.config.num_layers)]
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.audio_heads = [
            nn.Linear(self.config.hidden_size, self.config.audio_vocab_size, bias=False)
            for _ in range(self.config.num_codebooks - 1)
        ]

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"inputs_embeds must have shape [batch, length, {self.config.hidden_size}]"
            )
        length = inputs_embeds.shape[1]
        if length > self.config.max_position_embeddings:
            raise ValueError("depth sequence exceeds max_position_embeddings")
        positions = mx.arange(length, dtype=mx.int32)
        hidden_states = inputs_embeds + self.pos_embedding(positions)[None, :, :]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


def generate_depth_codes(
    language_model: LanguageModel,
    decoder: RVQDepthDecoder,
    last_hidden: mx.array,
    semantic_code: mx.array,
    key: mx.array,
    *,
    cfg_scale: float = 1.5,
    top_k: int = 50,
) -> tuple[mx.array, mx.array, mx.array]:
    """Generate residual RVQ codes for one conditional/unconditional frame pair."""
    if last_hidden.shape != (2, decoder.config.hidden_size):
        raise ValueError(f"last_hidden must have shape [2, {decoder.config.hidden_size}]")
    if semantic_code.shape != (2,):
        raise ValueError("semantic_code must have shape [2]")

    model_config = ModelConfig()
    sequence = [decoder.projection(last_hidden)[:, None, :]]
    semantic_embed = language_model.model.embed_tokens(
        semantic_code + model_config.audio_code_offset
    )
    sequence.append(decoder.projection(semantic_embed)[:, None, :])
    codes = [semantic_code]
    hidden_parts = []

    for index in range(1, decoder.config.num_codebooks):
        hidden = decoder(mx.concatenate(sequence, axis=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = decoder.audio_heads[index - 1](hidden).astype(mx.float32)
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * cfg_scale
        sampled, key = sample_top_k(guided, key, top_k=top_k)
        code = mx.repeat(sampled, 2, axis=0).astype(mx.int32)
        codes.append(code)
        if index < decoder.config.num_codebooks - 1:
            embedding_id = code + (index - 1) * decoder.config.audio_vocab_size
            embedded = decoder.audio_embeddings(embedding_id)
            sequence.append(decoder.projection(embedded)[:, None, :])

    return mx.stack(codes, axis=1), mx.concatenate(hidden_parts, axis=-1), key


def embed_audio_frame(
    language_model: LanguageModel,
    decoder: RVQDepthDecoder,
    frame_codes: mx.array,
) -> mx.array:
    if frame_codes.shape != (2, decoder.config.num_codebooks):
        raise ValueError(f"frame_codes must have shape [2, {decoder.config.num_codebooks}]")
    model_config = ModelConfig()
    semantic = language_model.model.embed_tokens(
        frame_codes[:, :1] + model_config.audio_code_offset
    )
    offsets = mx.arange(decoder.config.num_codebooks - 1, dtype=mx.int32)
    offsets = offsets * decoder.config.audio_vocab_size
    residual = decoder.audio_embeddings(frame_codes[:, 1:] + offsets[None, :])
    return (semantic + mx.sum(residual, axis=1, keepdims=True)) * (
        decoder.config.num_codebooks**-0.5
    )

