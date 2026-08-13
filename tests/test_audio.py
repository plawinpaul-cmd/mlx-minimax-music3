import io

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from mlx_minimax_music3.audio import audio_numpy, wav_bytes, write_wav


def test_audio_numpy_transposes_and_clips() -> None:
    audio = mx.array([[[2.0, -2.0], [0.5, -0.5]]])
    values = audio_numpy(audio)
    np.testing.assert_array_equal(values, [[1.0, 0.5], [-1.0, -0.5]])


def test_wav_bytes_are_pcm16_stereo() -> None:
    audio = mx.zeros((1, 2, 100))
    payload = wav_bytes(audio, 44_100)
    values, rate = sf.read(io.BytesIO(payload), always_2d=True)

    assert rate == 44_100
    assert values.shape == (100, 2)


def test_write_wav_creates_parent_directories(tmp_path) -> None:
    target = write_wav(tmp_path / "nested" / "out.wav", mx.zeros((1, 2, 10)), 44_100)
    assert target.is_file()


def test_invalid_audio_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        audio_numpy(mx.zeros((2, 10)))

