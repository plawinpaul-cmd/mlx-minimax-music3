import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from mlx_minimax_music3.language_model import LanguageModel, LanguageModelConfig


def tiny_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=32,
        rope_theta=10_000,
    )


def test_language_model_preserves_official_parameter_tree() -> None:
    model = LanguageModel(tiny_config())
    flat = dict(model.parameters().items())

    assert "model" in flat
    assert "lm_head" in flat
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (8, 8)


def test_cached_hidden_state_matches_full_sequence() -> None:
    mx.random.seed(11)
    model = LanguageModel(tiny_config())
    ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    full = model.hidden_states(ids)

    cache = [KVCache() for _ in model.layers]
    model.hidden_states(ids[:, :2], cache=cache)
    cached_last = model.hidden_states(ids[:, 2:], cache=cache)
    mx.eval(full, cached_last)

    np.testing.assert_allclose(
        np.asarray(cached_last[:, -1]), np.asarray(full[:, -1]), atol=1e-5, rtol=1e-5
    )


def test_hidden_state_api_requires_exactly_one_input() -> None:
    model = LanguageModel(tiny_config())
    with pytest.raises(ValueError):
        model.hidden_states()
    with pytest.raises(ValueError):
        model.hidden_states(mx.array([[1]]), input_embeddings=mx.zeros((1, 1, 8)))

