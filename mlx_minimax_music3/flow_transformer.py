from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class FlowTransformerConfig:
    in_channels: int = 128
    condition_dim: int = 2048
    num_layers: int = 36
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    ff_inner_dim: int = 8192
    rotary_dim: int = 32
    fourier_embedding_dim: int = 256

    def validate(self) -> None:
        if self.num_layers < 1 or self.num_attention_heads < 1:
            raise ValueError("transformer layer and head counts must be positive")
        if self.rotary_dim < 0 or self.rotary_dim > self.attention_head_dim:
            raise ValueError("rotary_dim must fit within attention_head_dim")
        if self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even")
        if self.fourier_embedding_dim % 2:
            raise ValueError("fourier_embedding_dim must be even")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def concat_channels(self) -> int:
        return 2 * self.in_channels + self.condition_dim


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even")
        self.weight = mx.random.normal((embedding_dim // 2, 1))

    def __call__(self, timestep: mx.array) -> mx.array:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [batch]")
        angles = 2.0 * math.pi * timestep[:, None] @ self.weight.T
        return mx.concatenate((mx.cos(angles), mx.sin(angles)), axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(hidden_states)))


def rotary_frequencies(
    sequence_length: int,
    rotary_dim: int,
    theta: float = 10_000.0,
) -> tuple[mx.array, mx.array]:
    if sequence_length < 1 or rotary_dim < 0 or rotary_dim % 2:
        raise ValueError("sequence_length must be positive and rotary_dim must be non-negative even")
    if rotary_dim == 0:
        empty = mx.zeros((sequence_length, 0), dtype=mx.float32)
        return empty, empty
    inv_freq = 1.0 / (
        theta
        ** (
            mx.arange(0, rotary_dim, 2, dtype=mx.float32)
            / rotary_dim
        )
    )
    steps = mx.arange(sequence_length, dtype=mx.float32)
    frequencies = steps[:, None] * inv_freq[None, :]
    frequencies = mx.concatenate((frequencies, frequencies), axis=-1)
    return mx.cos(frequencies), mx.sin(frequencies)


def apply_partial_rotary(
    hidden_states: mx.array,
    rotary: tuple[mx.array, mx.array],
) -> mx.array:
    """Apply reference partial RoPE to `[batch, length, heads, dim]`."""
    if hidden_states.ndim != 4:
        raise ValueError("hidden_states must have shape [batch, length, heads, dim]")
    cos, sin = rotary
    rotary_dim = cos.shape[-1]
    if cos.shape != sin.shape or cos.shape[0] != hidden_states.shape[1]:
        raise ValueError("rotary frequencies do not match the sequence")
    if rotary_dim == 0:
        return hidden_states
    if rotary_dim > hidden_states.shape[-1] or rotary_dim % 2:
        raise ValueError("invalid rotary dimension")

    rotated = hidden_states[..., :rotary_dim]
    half = rotary_dim // 2
    rotate_half = mx.concatenate((-rotated[..., half:], rotated[..., :half]), axis=-1)
    cos = cos[None, :, None, :].astype(hidden_states.dtype)
    sin = sin[None, :, None, :].astype(hidden_states.dtype)
    leading = rotated * cos + rotate_half * sin
    return mx.concatenate((leading, hidden_states[..., rotary_dim:]), axis=-1)


class FlowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.scale = head_dim**-0.5
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = [nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)]

    def __call__(
        self,
        hidden_states: mx.array,
        rotary: tuple[mx.array, mx.array],
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = apply_partial_rotary(self.to_q(hidden_states).reshape(shape), rotary)
        key = apply_partial_rotary(self.to_k(hidden_states).reshape(shape), rotary)
        value = self.to_v(hidden_states).reshape(shape)
        output = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.scale,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, self.inner_dim)
        return self.to_out[1](self.to_out[0](output))


class FlowTransformerBlock(nn.Module):
    def __init__(self, config: FlowTransformerConfig):
        super().__init__()
        dim = config.inner_dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlowAttention(
            dim,
            config.num_attention_heads,
            config.attention_head_dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, config.ff_inner_dim * 2)
        self.ff_out = nn.Linear(config.ff_inner_dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary: tuple[mx.array, mx.array],
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary)
        gate_states, gate = mx.split(self.ff_in(self.norm2(hidden_states)), 2, axis=-1)
        return hidden_states + self.ff_out(gate_states * nn.silu(gate))


class FlowTransformer(nn.Module):
    """Music3 flow transformer using channel-last MLX sequences."""

    def __init__(self, config: FlowTransformerConfig | None = None):
        super().__init__()
        self.config = config or FlowTransformerConfig()
        self.config.validate()
        self.time_proj = FourierEmbedding(self.config.fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(
            self.config.fourier_embedding_dim,
            self.config.inner_dim,
        )
        self.preprocess_conv = nn.Conv1d(
            self.config.concat_channels,
            self.config.concat_channels,
            kernel_size=1,
            bias=False,
        )
        self.proj_in = nn.Linear(self.config.concat_channels, self.config.inner_dim, bias=False)
        self.transformer_blocks = [
            FlowTransformerBlock(self.config) for _ in range(self.config.num_layers)
        ]
        self.proj_out = nn.Linear(self.config.inner_dim, self.config.in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(
            self.config.in_channels,
            self.config.in_channels,
            kernel_size=1,
            bias=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, length, channels]")
        batch, length, channels = hidden_states.shape
        if channels != self.config.in_channels:
            raise ValueError(f"expected {self.config.in_channels} latent channels, got {channels}")
        expected_condition = (batch, length, self.config.condition_dim)
        if encoder_hidden_states.shape != expected_condition:
            raise ValueError(
                f"encoder_hidden_states must have shape {expected_condition}, got {encoder_hidden_states.shape}"
            )
        if timestep.shape != (batch,):
            raise ValueError(f"timestep must have shape [{batch}]")

        zeros = mx.zeros_like(hidden_states)
        hidden_states = mx.concatenate(
            (hidden_states, zeros, encoder_hidden_states), axis=-1
        )
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        time_embedding = self.time_embed(self.time_proj(timestep))
        hidden_states = self.proj_in(hidden_states)
        hidden_states = mx.concatenate((time_embedding[:, None, :], hidden_states), axis=1)
        rotary = rotary_frequencies(
            hidden_states.shape[1],
            self.config.rotary_dim,
        )
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary)
        hidden_states = self.proj_out(hidden_states[:, 1:])
        return self.postprocess_conv(hidden_states) + hidden_states

