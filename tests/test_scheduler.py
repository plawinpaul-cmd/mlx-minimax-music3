import mlx.core as mx
import numpy as np
import pytest

from mlx_minimax_music3.scheduler import (
    blend_overlap,
    carry_window,
    chunk_starts,
    euler_step,
    flow_timesteps,
    restore_overlap,
    waveform_crop,
)


def test_flow_schedule_and_euler_steps_reach_unit_time() -> None:
    times = flow_timesteps(4)
    sample = mx.zeros((1, 2, 1))
    velocity = mx.ones_like(sample)
    for _ in range(4):
        sample = euler_step(sample, velocity, 4)
    mx.eval(times, sample)

    np.testing.assert_allclose(np.asarray(times), [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(np.asarray(sample), 1.0)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, [0]), (200, [0]), (201, [0, 100]), (300, [0, 100]), (301, [0, 100, 200])],
)
def test_chunk_starts_match_reference(frames: int, expected: list[int]) -> None:
    assert chunk_starts(frames) == expected


def test_overlap_blend_and_restore_match_reference_contract() -> None:
    latents = mx.zeros((1, 4, 1))
    noise = mx.ones((1, 2, 1)) * 2
    previous = mx.ones((1, 2, 1)) * 6
    blended = blend_overlap(latents, noise, previous, overlap=2, timestep=0.5)
    restored = restore_overlap(blended, previous, overlap=2)
    mx.eval(blended, restored)

    np.testing.assert_allclose(np.asarray(blended)[0, :2, 0], [4.000001, 4.000001])
    np.testing.assert_array_equal(np.asarray(restored)[0, :2, 0], [6, 6])


def test_carry_window_uses_reference_slice() -> None:
    latents = mx.arange(500).reshape(1, 500, 1)
    carry = carry_window(latents)
    mx.eval(carry)
    assert carry.shape == (1, 172, 1)
    assert carry[0, 0, 0].item() == 156
    assert carry[0, -1, 0].item() == 327


def test_waveform_crop_matches_reference_constants() -> None:
    assert waveform_crop(0, 3).left == 0
    assert waveform_crop(0, 3).right == 258 * 512
    assert waveform_crop(1, 3).left == 86 * 512
    assert waveform_crop(2, 3).right == 0
