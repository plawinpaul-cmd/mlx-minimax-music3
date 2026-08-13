import mlx.core as mx
import numpy as np
import pytest

from mlx_minimax_music3.condition_encoder import (
    ConditionEncoder,
    ConditionEncoderConfig,
    nearest_resize_1d,
)


def test_default_condition_length_matches_reference() -> None:
    assert ConditionEncoderConfig().output_length(5) == 17


def test_nearest_resize_matches_reference_index_rule() -> None:
    inputs = mx.array([[[10.0], [20.0], [30.0]]])
    output = nearest_resize_1d(inputs, 5)
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output)[0, :, 0], [10, 10, 20, 20, 30])


def test_condition_encoder_mixes_projects_and_resizes() -> None:
    config = ConditionEncoderConfig(
        condition_hidden_dim=2,
        num_condition_layers=2,
        out_dim=2,
        input_sampling_rate=1,
        input_hop_length=1,
        output_sampling_rate=2,
        output_hop_length=1,
    )
    model = ConditionEncoder(config)
    weight = np.zeros((2, 3, 2), dtype=np.float32)
    weight[:, 1, :] = np.eye(2, dtype=np.float32)
    model.proj.weight = mx.array(weight)
    model.proj.bias = mx.zeros((2,))

    # Each frame contains two hidden-state groups; zero logits mix them equally.
    inputs = mx.array([[[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]]])
    output = model(inputs)
    mx.eval(output)

    expected = np.array([[[4, 6], [4, 6], [12, 14], [12, 14]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(output), expected, atol=1e-6)


def test_condition_encoder_validates_feature_count() -> None:
    model = ConditionEncoder(
        ConditionEncoderConfig(condition_hidden_dim=2, num_condition_layers=2, out_dim=2)
    )
    with pytest.raises(ValueError, match="expected 4"):
        model(mx.zeros((1, 3, 5)))

