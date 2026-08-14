import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_minimax_music3.flow_transformer import (
    FlowTransformer,
    FlowTransformerConfig,
    FourierEmbedding,
    apply_partial_rotary,
    rotary_frequencies,
)


def tiny_config() -> FlowTransformerConfig:
    return FlowTransformerConfig(
        in_channels=4,
        condition_dim=6,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=4,
        ff_inner_dim=12,
        rotary_dim=2,
        fourier_embedding_dim=4,
    )


def test_fourier_embedding_uses_trained_projection_weight() -> None:
    embedding = FourierEmbedding(4)
    embedding.weight = mx.array([[1.0], [2.0]])
    output = embedding(mx.array([0.0, 0.25]))
    mx.eval(output)

    expected = np.array(
        [[1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0]], dtype=np.float32
    )
    np.testing.assert_allclose(np.asarray(output), expected, atol=1e-6)


def test_partial_rotary_preserves_unrotated_suffix() -> None:
    hidden = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = mx.array([[0.0, 0.0]])
    sin = mx.array([[1.0, 1.0]])
    output = apply_partial_rotary(hidden, (cos, sin))
    mx.eval(output)

    np.testing.assert_allclose(np.asarray(output), [[[[-2.0, 1.0, 3.0, 4.0]]]])


def test_rotary_frequency_zero_position_is_identity() -> None:
    cos, sin = rotary_frequencies(3, 4)
    mx.eval(cos, sin)
    np.testing.assert_array_equal(np.asarray(cos[0]), np.ones(4))
    np.testing.assert_array_equal(np.asarray(sin[0]), np.zeros(4))


def test_rotary_frequencies_are_cached_by_shape() -> None:
    rotary_frequencies.cache_clear()
    first = rotary_frequencies(7, 4)
    second = rotary_frequencies(7, 4)

    assert first[0] is second[0]
    assert rotary_frequencies.cache_info().hits == 1


def test_flow_transformer_shape_and_finite_output() -> None:
    mx.random.seed(5)
    model = FlowTransformer(tiny_config())
    latents = mx.random.normal((1, 7, 4))
    condition = mx.random.normal((1, 7, 6))
    output = model(latents, mx.array([0.5]), condition)
    mx.eval(output)

    assert output.shape == latents.shape
    assert np.isfinite(np.asarray(output)).all()


def test_flow_parameter_tree_matches_diffusers_names() -> None:
    model = FlowTransformer(tiny_config())
    parameters = model.parameters()

    assert parameters["time_proj"]["weight"].shape == (2, 1)
    assert parameters["time_embed"]["linear_1"]["weight"].shape == (8, 4)
    assert parameters["transformer_blocks"][0]["attn"]["to_out"][0]["weight"].shape == (8, 8)
    assert parameters["preprocess_conv"]["weight"].shape == (14, 1, 14)


def test_flow_transformer_validates_shapes() -> None:
    model = FlowTransformer(tiny_config())
    with pytest.raises(ValueError, match="latent channels"):
        model(mx.zeros((1, 2, 3)), mx.array([0.0]), mx.zeros((1, 2, 6)))
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        model(mx.zeros((1, 2, 4)), mx.array([0.0]), mx.zeros((1, 3, 6)))


def test_invalid_rotary_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        FlowTransformer(FlowTransformerConfig(attention_head_dim=4, rotary_dim=3))
