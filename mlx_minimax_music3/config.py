from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _environment_flag(
    values: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    value = values.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


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
class OptimizationConfig:
    """Independent runtime optimizations with environment-variable kill switches."""

    pruned_semantic_head: bool = True
    depth_kv_cache: bool = True
    batched_dit_cfg: bool = True
    compiled_dit: bool = True

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str] | None = None,
    ) -> "OptimizationConfig":
        values = os.environ if values is None else values
        return cls(
            pruned_semantic_head=_environment_flag(
                values,
                "MLX_MUSIC3_PRUNED_HEAD",
                default=True,
            ),
            depth_kv_cache=_environment_flag(
                values,
                "MLX_MUSIC3_DEPTH_KV_CACHE",
                default=True,
            ),
            batched_dit_cfg=_environment_flag(
                values,
                "MLX_MUSIC3_BATCHED_DIT_CFG",
                default=True,
            ),
            compiled_dit=_environment_flag(
                values,
                "MLX_MUSIC3_COMPILED_DIT",
                default=True,
            ),
        )


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
