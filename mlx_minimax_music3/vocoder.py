from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VocoderConfig:
    latent_channels: int = 128
    decoder_input_dim: int = 1024
    decoder_hidden_dim: int = 1536
    upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2)
    sampling_rate: int = 44_100

    def validate(self) -> None:
        if self.latent_channels < 2 or self.latent_channels % 2:
            raise ValueError("latent_channels must be positive and even")
        if self.decoder_input_dim < 1 or self.decoder_hidden_dim < 1:
            raise ValueError("decoder dimensions must be positive")
        if not self.upsampling_ratios or any(ratio < 1 for ratio in self.upsampling_ratios):
            raise ValueError("upsampling ratios must be positive")
        divisor = 2 ** len(self.upsampling_ratios)
        if self.decoder_hidden_dim % divisor:
            raise ValueError("decoder_hidden_dim must be divisible by 2 ** number of blocks")

    @property
    def hop_length(self) -> int:
        result = 1
        for ratio in self.upsampling_ratios:
            result *= ratio
        return result


def fold_weight_norm(
    weight_g: mx.array,
    weight_v: mx.array,
    dim: int = 0,
    eps: float = 1e-12,
) -> mx.array:
    """Fold PyTorch legacy weight normalization into an ordinary kernel."""
    if weight_g.ndim != weight_v.ndim:
        raise ValueError("weight_g and weight_v must have the same rank")
    if dim < 0:
        dim += weight_v.ndim
    if dim < 0 or dim >= weight_v.ndim:
        raise ValueError("dim is outside the weight rank")
    expected = list(weight_v.shape)
    for axis in range(weight_v.ndim):
        if axis != dim:
            expected[axis] = 1
    if tuple(weight_g.shape) != tuple(expected):
        raise ValueError(f"expected weight_g shape {tuple(expected)}, got {weight_g.shape}")

    axes = tuple(axis for axis in range(weight_v.ndim) if axis != dim)
    norm = mx.sqrt(mx.sum(weight_v.astype(mx.float32) ** 2, axis=axes, keepdims=True))
    return weight_v * (weight_g / mx.maximum(norm.astype(weight_g.dtype), eps))


def pytorch_conv1d_to_mlx(weight: mx.array) -> mx.array:
    """Convert `[out, in, kernel]` to MLX `[out, kernel, in]`."""
    if weight.ndim != 3:
        raise ValueError("Conv1d weight must have rank 3")
    return weight.transpose(0, 2, 1)


def pytorch_conv_transpose1d_to_mlx(weight: mx.array) -> mx.array:
    """Convert `[in, out, kernel]` to MLX `[out, kernel, in]`."""
    if weight.ndim != 3:
        raise ValueError("ConvTranspose1d weight must have rank 3")
    return weight.transpose(1, 2, 0)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        alpha = self.alpha.astype(hidden_states.dtype)
        return hidden_states + mx.sin(alpha * hidden_states) ** 2 / (alpha + 1e-9)


class VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        padding = (7 - 1) * dilation // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            dilation=dilation,
            padding=padding,
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = VocoderResidualUnit(output_dim, dilation=9)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class Vocoder(nn.Module):
    """Decode channel-last Flow-VAE latents to stereo waveform samples."""

    def __init__(self, config: VocoderConfig | None = None):
        super().__init__()
        self.config = config or VocoderConfig()
        self.config.validate()
        self.dec_in_proj = nn.Conv1d(
            self.config.latent_channels // 2,
            self.config.decoder_input_dim,
            kernel_size=1,
        )
        self.conv_in = nn.Conv1d(
            self.config.decoder_input_dim,
            self.config.decoder_hidden_dim,
            kernel_size=7,
            padding=3,
        )
        self.blocks = []
        output_dim = self.config.decoder_hidden_dim
        for index, stride in enumerate(self.config.upsampling_ratios):
            input_dim = self.config.decoder_hidden_dim // (2**index)
            output_dim = self.config.decoder_hidden_dim // (2 ** (index + 1))
            self.blocks.append(VocoderBlock(input_dim, output_dim, stride))
        self.snake_out = Snake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def __call__(self, latents: mx.array) -> mx.array:
        if latents.ndim != 3 or latents.shape[-1] != self.config.latent_channels:
            raise ValueError(
                f"latents must have shape [batch, length, {self.config.latent_channels}]"
            )
        batch, length, _ = latents.shape
        hidden_states = latents.reshape(
            batch,
            length,
            2,
            self.config.latent_channels // 2,
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch * 2,
            length,
            self.config.latent_channels // 2,
        )
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = mx.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(batch, 2, waveform.shape[1])

