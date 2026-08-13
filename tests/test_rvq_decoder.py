import mlx.core as mx
import numpy as np
import pytest

from mlx_minimax_music3.language_model import LanguageModel, LanguageModelConfig
from mlx_minimax_music3.rvq_decoder import (
    RVQDecoderConfig,
    RVQDepthDecoder,
    embed_audio_frame,
    generate_depth_codes,
)


def tiny_models() -> tuple[LanguageModel, RVQDepthDecoder]:
    language = LanguageModel(
        LanguageModelConfig(
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=200_000,
            max_position_embeddings=32,
            rope_theta=10_000,
        )
    )
    decoder = RVQDepthDecoder(
        RVQDecoderConfig(
            hidden_size=8,
            num_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            audio_vocab_size=16,
            num_codebooks=4,
            max_position_embeddings=8,
        )
    )
    return language, decoder


def test_depth_attention_is_causal() -> None:
    _, decoder = tiny_models()
    first = mx.zeros((1, 3, 8))
    second = mx.concatenate((first[:, :2], mx.ones((1, 1, 8))), axis=1)
    first_out = decoder(first)
    second_out = decoder(second)
    mx.eval(first_out, second_out)

    np.testing.assert_allclose(
        np.asarray(first_out[:, :2]), np.asarray(second_out[:, :2]), atol=1e-6, rtol=1e-6
    )


def test_generate_depth_codes_and_frame_embedding() -> None:
    language, decoder = tiny_models()
    last_hidden = mx.zeros((2, 8))
    semantic_code = mx.array([3, 3], dtype=mx.int32)
    codes, hidden, _ = generate_depth_codes(
        language,
        decoder,
        last_hidden,
        semantic_code,
        mx.random.key(7),
        top_k=1,
    )
    feedback = embed_audio_frame(language, decoder, codes)
    mx.eval(codes, hidden, feedback)

    assert codes.shape == (2, 4)
    assert hidden.shape == (1, 3 * 8)
    assert feedback.shape == (2, 1, 8)
    assert np.isfinite(np.asarray(hidden)).all()
    assert np.isfinite(np.asarray(feedback)).all()


def test_depth_decoder_rejects_too_long_sequence() -> None:
    _, decoder = tiny_models()
    with pytest.raises(ValueError, match="exceeds"):
        decoder(mx.zeros((1, 9, 8)))
