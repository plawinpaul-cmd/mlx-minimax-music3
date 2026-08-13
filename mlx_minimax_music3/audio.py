from __future__ import annotations

import io
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf


def audio_numpy(audio: mx.array) -> np.ndarray:
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 2:
        raise ValueError("audio must have shape [1, 2, samples]")
    mx.eval(audio)
    values = np.asarray(audio[0].astype(mx.float32)).T
    if not np.isfinite(values).all():
        raise ValueError("audio contains non-finite samples")
    return np.clip(values, -1.0, 1.0)


def write_wav(path: str | Path, audio: mx.array, sampling_rate: int) -> Path:
    if sampling_rate < 1:
        raise ValueError("sampling_rate must be positive")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio_numpy(audio), sampling_rate, format="WAV", subtype="PCM_16")
    return target


def wav_bytes(audio: mx.array, sampling_rate: int) -> bytes:
    if sampling_rate < 1:
        raise ValueError("sampling_rate must be positive")
    buffer = io.BytesIO()
    sf.write(buffer, audio_numpy(audio), sampling_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()

