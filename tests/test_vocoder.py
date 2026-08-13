import mlx.core as mx
import numpy as np
import pytest

from mlx_minimax_music3.vocoder import (
    Vocoder,
    VocoderConfig,
    fold_weight_norm,
    pytorch_conv1d_to_mlx,
    pytorch_conv_transpose1d_to_mlx,
)


def tiny_config() -> VocoderConfig:
    return VocoderConfig(
        latent_channels=8,
        decoder_input_dim=16,
        decoder_hidden_dim=32,
        upsampling_ratios=(2, 2),
    )


def test_weight_norm_folds_per_dim_zero_slice() -> None:
    weight_v = mx.array([[[3.0, 4.0]], [[0.0, 5.0]]])
    weight_g = mx.array([[[10.0]], [[2.0]]])
    folded = fold_weight_norm(weight_g, weight_v)
    mx.eval(folded)

    expected = np.array([[[6.0, 8.0]], [[0.0, 2.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(folded), expected, atol=1e-6)


def test_kernel_layout_conversions() -> None:
    conv = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    conv_t = mx.arange(3 * 2 * 5).reshape(3, 2, 5)

    assert pytorch_conv1d_to_mlx(conv).shape == (2, 5, 3)
    assert pytorch_conv_transpose1d_to_mlx(conv_t).shape == (2, 5, 3)
    assert pytorch_conv1d_to_mlx(conv)[1, 4, 2].item() == conv[1, 2, 4].item()
    assert pytorch_conv_transpose1d_to_mlx(conv_t)[1, 4, 2].item() == conv_t[2, 1, 4].item()


def test_vocoder_outputs_stereo_at_hop_length() -> None:
    mx.random.seed(13)
    model = Vocoder(tiny_config())
    latents = mx.random.normal((1, 5, 8))
    waveform = model(latents)
    mx.eval(waveform)

    assert waveform.shape == (1, 2, 5 * model.config.hop_length)
    assert np.isfinite(np.asarray(waveform)).all()
    assert np.abs(np.asarray(waveform)).max() <= 1.0


def test_stereo_channels_use_separate_latent_halves() -> None:
    mx.random.seed(17)
    model = Vocoder(tiny_config())
    left_only = mx.zeros((1, 3, 8))
    left_only[:, :, :4] = 1.0
    right_only = mx.zeros((1, 3, 8))
    right_only[:, :, 4:] = 1.0
    left_output = model(left_only)
    right_output = model(right_only)
    mx.eval(left_output, right_output)

    # Both stereo rows share decoder weights; swapping latent halves swaps rows.
    np.testing.assert_allclose(
        np.asarray(left_output[:, 0]), np.asarray(right_output[:, 1]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(left_output[:, 1]), np.asarray(right_output[:, 0]), atol=1e-6
    )


def test_vocoder_rejects_wrong_latent_shape() -> None:
    model = Vocoder(tiny_config())
    with pytest.raises(ValueError, match="latents must have shape"):
        model(mx.zeros((1, 3, 7)))


def test_invalid_weight_norm_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight_g shape"):
        fold_weight_norm(mx.ones((2, 2, 1)), mx.ones((2, 3, 1)))
