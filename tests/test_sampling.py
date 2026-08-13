import mlx.core as mx
import numpy as np
import pytest

from mlx_minimax_music3.sampling import sample_top_k, semantic_guided_logits


def test_top_k_one_always_selects_the_maximum() -> None:
    logits = mx.array([[0.0, 2.0, 1.0]])
    sample, _ = sample_top_k(logits, mx.random.key(7), top_k=1)
    assert sample.item() == 1


def test_sampling_is_deterministic_for_a_fixed_key() -> None:
    logits = mx.array([[0.1, 0.2, 0.3, 0.4]])
    first, first_next = sample_top_k(logits, mx.random.key(123), top_k=4)
    second, second_next = sample_top_k(logits, mx.random.key(123), top_k=4)
    mx.eval(first, second, first_next, second_next)

    assert first.item() == second.item()
    np.testing.assert_array_equal(np.asarray(first_next), np.asarray(second_next))


def test_sampling_sanitizes_non_finite_logits() -> None:
    logits = mx.array([[mx.nan, mx.inf, -mx.inf]])
    sample, _ = sample_top_k(logits, mx.random.key(0), top_k=1)
    assert sample.item() == 1


def test_semantic_guidance_applies_top_k_and_vocab_mask() -> None:
    logits = mx.array([[1.0, 4.0, 3.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
    allowed = mx.array([True, True, False, True])
    guided = semantic_guided_logits(logits, allowed, cfg_scale=1.5, conditional_top_k=2)
    mx.eval(guided)

    values = np.asarray(guided)
    assert np.isneginf(values[0, 0])
    assert values[0, 1] == pytest.approx(6.0)
    assert np.isneginf(values[0, 2])
    assert np.isneginf(values[0, 3])


def test_semantic_guidance_validates_shapes() -> None:
    with pytest.raises(ValueError):
        semantic_guided_logits(mx.zeros((1, 4)), mx.ones((4,), dtype=mx.bool_))
