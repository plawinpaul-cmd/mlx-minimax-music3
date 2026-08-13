from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ConditionEncoderConfig:
    condition_hidden_dim: int = 4096
    num_condition_layers: int = 8
    out_dim: int = 2048
    input_sampling_rate: int = 24_000
    input_hop_length: int = 960
    output_sampling_rate: int = 44_100
    output_hop_length: int = 512

    def output_length(self, num_frames: int) -> int:
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        return max(
            1,
            int(
                num_frames
                * self.output_sampling_rate
                / self.input_sampling_rate
                * self.input_hop_length
                / self.output_hop_length
            ),
        )


def nearest_resize_1d(hidden_states: mx.array, output_length: int) -> mx.array:
    """Resize a channel-last sequence with PyTorch-compatible nearest indices."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, length, channels]")
    input_length = hidden_states.shape[1]
    if input_length < 1 or output_length < 1:
        raise ValueError("input and output lengths must be positive")
    if input_length == output_length:
        return hidden_states

    indices = mx.floor(
        mx.arange(output_length, dtype=mx.float32) * input_length / output_length
    ).astype(mx.int32)
    return mx.take(hidden_states, indices, axis=1)


class ConditionEncoder(nn.Module):
    """Project AR hidden states onto the Flow-VAE latent timeline."""

    def __init__(self, config: ConditionEncoderConfig | None = None):
        super().__init__()
        self.config = config or ConditionEncoderConfig()
        self.layer_weight_logits = mx.zeros((self.config.num_condition_layers,))
        self.layer_scale = mx.ones((1,))
        self.proj = nn.Conv1d(
            self.config.condition_hidden_dim,
            self.config.out_dim,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, frames, features]")
        batch, frames, features = hidden_states.shape
        expected = self.config.num_condition_layers * self.config.condition_hidden_dim
        if features != expected:
            raise ValueError(f"expected {expected} hidden features, got {features}")

        grouped = hidden_states.reshape(
            batch,
            frames,
            self.config.num_condition_layers,
            self.config.condition_hidden_dim,
        )
        weights = mx.softmax(self.layer_weight_logits, axis=0).astype(grouped.dtype)
        mixed = mx.sum(grouped * weights[None, None, :, None], axis=2)
        projected = self.proj(mixed * self.layer_scale.astype(mixed.dtype))
        return nearest_resize_1d(projected, self.config.output_length(frames))

