from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENT_LENGTH = 172
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 258


def flow_timesteps(num_inference_steps: int) -> mx.array:
    """Return the Music3 flow times: 0, 1/N, ..., (N-1)/N."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    return mx.arange(num_inference_steps, dtype=mx.float32) / num_inference_steps


def euler_step(sample: mx.array, velocity: mx.array, num_inference_steps: int) -> mx.array:
    if sample.shape != velocity.shape:
        raise ValueError("sample and velocity must have the same shape")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    return (sample.astype(mx.float32) + velocity.astype(mx.float32) / num_inference_steps).astype(
        velocity.dtype
    )


def chunk_starts(num_frames: int) -> list[int]:
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if num_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


def blend_overlap(
    latents: mx.array,
    noise_prompt: mx.array,
    previous_latent: mx.array,
    overlap: int,
    timestep: mx.array | float,
) -> mx.array:
    if overlap < 0 or overlap > latents.shape[1]:
        raise ValueError("overlap is outside the latent sequence")
    if overlap == 0:
        return latents
    if noise_prompt.shape[1] < overlap or previous_latent.shape[1] < overlap:
        raise ValueError("overlap prompts are shorter than overlap")

    time = mx.asarray(timestep, dtype=latents.dtype)
    prefix = (
        (1.0 - (1.0 - 1e-6) * time) * noise_prompt[:, :overlap]
        + time * previous_latent[:, :overlap]
    )
    return mx.concatenate((prefix, latents[:, overlap:]), axis=1)


def restore_overlap(latents: mx.array, previous_latent: mx.array, overlap: int) -> mx.array:
    if overlap == 0:
        return latents
    if overlap < 0 or overlap > latents.shape[1] or previous_latent.shape[1] < overlap:
        raise ValueError("invalid overlap")
    return mx.concatenate((previous_latent[:, :overlap], latents[:, overlap:]), axis=1)


def carry_window(latents: mx.array) -> mx.array:
    start = max(0, latents.shape[1] - 2 * OVERLAP_LATENT_LENGTH)
    end = max(start, latents.shape[1] - OVERLAP_LATENT_LENGTH)
    return latents[:, start:end]


@dataclass(frozen=True)
class WaveformCrop:
    left: int
    right: int


def waveform_crop(chunk_index: int, num_chunks: int, hop_length: int = 512) -> WaveformCrop:
    if num_chunks < 1 or chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError("invalid chunk index")
    if hop_length < 1:
        raise ValueError("hop_length must be positive")
    return WaveformCrop(
        left=0 if chunk_index == 0 else CROP_LEFT_LATENT * hop_length,
        right=0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT * hop_length,
    )

