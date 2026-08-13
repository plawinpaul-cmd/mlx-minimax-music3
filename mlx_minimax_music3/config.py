from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    frame_rate: int = 25
    num_codebooks: int = 8
    semantic_vocab_size: int = 16_384
    audio_vocab_size: int = 1_024
    audio_code_offset: int = 151_675
    audio_end_token_id: int = 151_670
    audio_cfg_token_id: int = 151_654
    max_prompt_tokens: int = 5_000
    max_audio_frames: int = 9_000
    sampling_rate: int = 44_100
    latent_hop_length: int = 512
    latent_channels: int = 128


@dataclass(frozen=True)
class GenerationConfig:
    audio_duration: float = 60.0
    seed: int = 0
    num_inference_steps: int = 30
    ar_cfg_scale: float = 1.5
    flow_cfg_scale: float = 1.7
    top_k: int = 50

    def max_frames(self, model: ModelConfig) -> int:
        if self.audio_duration <= 0:
            raise ValueError("audio_duration must be positive")
        frames = int(self.audio_duration * model.frame_rate)
        if frames < 1:
            raise ValueError("audio_duration is shorter than one audio frame")
        return min(frames, model.max_audio_frames)

    def validate(self) -> None:
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

