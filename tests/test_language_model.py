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


def test_pruned_semantic_head_matches_selected_full_logits() -> None:
    mx.random.seed(17)
    model = LanguageModel(tiny_config())
    hidden = mx.random.normal((2, 8))
    full_logits = model.logits(hidden)
    expected = mx.concatenate((full_logits[:, 12:17], full_logits[:, 3:4]), axis=-1)

    model.prepare_semantic_head(
        audio_code_offset=12,
        semantic_vocab_size=5,
        audio_end_token_id=3,
    )
    compact_logits = model.logits(hidden)
    mx.eval(expected, compact_logits)

    assert model.semantic_head_layout == (12, 5, 3)
    assert model.lm_head is None
    np.testing.assert_array_equal(np.asarray(compact_logits), np.asarray(expected))


def test_pruned_semantic_head_rejects_layout_change() -> None:
    model = LanguageModel(tiny_config())
    model.prepare_semantic_head(12, 5, 3)

    with pytest.raises(ValueError, match="different layout"):
        model.prepare_semantic_head(13, 5, 3)
