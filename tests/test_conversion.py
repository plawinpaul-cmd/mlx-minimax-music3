import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_minimax_music3.checkpoint import apply_component_quantization, load_component
from mlx_minimax_music3.conversion import (
    convert_component,
    convert_component_source_shard,
    convert_tensor,
    convert_vocoder_weights,
)
from mlx_minimax_music3.language_model import LanguageModel, LanguageModelConfig


def small_language_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=32,
        rope_theta=10_000,
    )


def test_quantized_tensor_emits_mlx_triplet() -> None:
    weight = mx.arange(64 * 64).reshape(64, 64).astype(mx.bfloat16)
    converted = convert_tensor(
        "language_model", "model.layers.0.self_attn.q_proj.weight", weight
    )

    assert set(converted) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.biases",
    }
    assert converted["model.layers.0.self_attn.q_proj.weight"].dtype == mx.uint32


def test_vocoder_conversion_folds_and_transposes() -> None:
    source = {
        "conv_in.weight_g": mx.array([[[10.0]], [[2.0]]]),
        "conv_in.weight_v": mx.array([[[3.0, 4.0]], [[0.0, 5.0]]]),
        "conv_in.bias": mx.zeros((2,)),
        "snake_out.alpha": mx.ones((1, 2, 1)),
    }
    converted = dict(convert_vocoder_weights(source))
    mx.eval(converted)

    assert set(converted) == {"conv_in.weight", "conv_in.bias", "snake_out.alpha"}
    assert converted["conv_in.weight"].shape == (2, 2, 1)
    np.testing.assert_allclose(
        np.asarray(converted["conv_in.weight"].astype(mx.float32)),
        np.array([[[6.0], [8.0]], [[0.0], [2.0]]]),
        atol=1e-2,
    )
    assert converted["snake_out.alpha"].shape == (1, 1, 2)


def test_language_conversion_round_trip_strict_load(tmp_path) -> None:
    mx.random.seed(31)
    config = small_language_config()
    source_model = LanguageModel(config)
    source_dir = tmp_path / "source" / "language_model"
    source_dir.mkdir(parents=True)
    source_weights = {
        key: value.astype(mx.bfloat16)
        for key, value in tree_flatten(source_model.parameters())
    }
    mx.save_safetensors(source_dir / "model.safetensors", source_weights)

    manifest = convert_component(
        tmp_path / "source",
        tmp_path / "target",
        "language_model",
        shard_size=40_000,
    )
    assert manifest["status"] == "complete"
    assert manifest["processed_source_files"] == ["model.safetensors"]
    assert len(manifest["shards"]) > 1

    converted = load_component(
        "language_model",
        LanguageModel(config),
        tmp_path / "target",
    )
    reference = LanguageModel(config)
    reference.load_weights(list(source_weights.items()), strict=True)
    apply_component_quantization("language_model", reference)
    mx.eval(reference.parameters(), converted.parameters())

    ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected = reference(ids)
    actual = converted(ids)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )


def test_complete_conversion_resumes_without_rewriting(tmp_path) -> None:
    source_dir = tmp_path / "source" / "condition_encoder"
    source_dir.mkdir(parents=True)
    mx.save_safetensors(
        source_dir / "diffusion_pytorch_model.safetensors",
        {
            "proj.weight": mx.ones((2, 2, 3)),
            "proj.bias": mx.zeros((2,)),
        },
    )
    first = convert_component(tmp_path / "source", tmp_path / "target", "condition_encoder")
    shard = tmp_path / "target" / "condition_encoder" / first["shards"][0]["file"]
    before = shard.stat().st_mtime_ns

    second = convert_component(tmp_path / "source", tmp_path / "target", "condition_encoder")
    assert second == first
    assert shard.stat().st_mtime_ns == before


def test_conversion_index_covers_every_manifest_tensor(tmp_path) -> None:
    source_dir = tmp_path / "source" / "condition_encoder"
    source_dir.mkdir(parents=True)
    mx.save_safetensors(
        source_dir / "diffusion_pytorch_model.safetensors",
        {"proj.weight": mx.ones((2, 2, 3)), "proj.bias": mx.zeros((2,))},
    )
    manifest = convert_component(tmp_path / "source", tmp_path / "target", "condition_encoder")
    index_path = tmp_path / "target" / "condition_encoder" / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())

    assert index["weight_map"] == dict(sorted(manifest["weight_map"].items()))


def test_component_can_convert_one_source_shard_at_a_time(tmp_path) -> None:
    source_dir = tmp_path / "source" / "language_model"
    source_dir.mkdir(parents=True)
    mx.save_safetensors(source_dir / "part-a.safetensors", {"model.norm.weight": mx.ones((64,))})
    mx.save_safetensors(source_dir / "part-b.safetensors", {"lm_head.weight": mx.ones((128, 64))})
    index = {
        "weight_map": {
            "model.norm.weight": "part-a.safetensors",
            "lm_head.weight": "part-b.safetensors",
        }
    }
    (source_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    first = convert_component_source_shard(
        tmp_path / "source", tmp_path / "target", "language_model", "part-a.safetensors"
    )
    assert first["status"] == "in_progress"
    assert first["processed_source_files"] == ["part-a.safetensors"]
    assert not (tmp_path / "target" / "language_model" / "model.safetensors.index.json").exists()

    second = convert_component_source_shard(
        tmp_path / "source", tmp_path / "target", "language_model", "part-b.safetensors"
    )
    assert second["status"] == "complete"
    assert second["processed_source_files"] == ["part-a.safetensors", "part-b.safetensors"]
    assert (tmp_path / "target" / "language_model" / "model.safetensors.index.json").is_file()


def test_incremental_conversion_rejects_unindexed_source(tmp_path) -> None:
    source_dir = tmp_path / "source" / "transformer"
    source_dir.mkdir(parents=True)
    mx.save_safetensors(source_dir / "expected.safetensors", {"proj_out.weight": mx.ones((64, 64))})
    (source_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"proj_out.weight": "expected.safetensors"}})
    )
    with pytest.raises(ValueError, match="official index"):
        convert_component_source_shard(
            tmp_path / "source", tmp_path / "target", "transformer", "other.safetensors"
        )
