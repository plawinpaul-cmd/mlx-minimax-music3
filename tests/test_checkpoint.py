import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_minimax_music3.checkpoint import (
    QUANTIZATION,
    apply_component_quantization,
    load_component_weights,
    read_checkpoint_config,
    write_checkpoint_config,
)


class QuantizableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [nn.Module()]
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.q_proj = nn.Linear(64, 32, bias=False)
        self.lm_head = nn.Linear(64, 128, bias=False)


def test_language_quantization_selects_backbone_but_not_output_head() -> None:
    model = QuantizableModel()
    selected = apply_component_quantization("language_model", model)

    assert selected == ["model.layers.0.self_attn.q_proj"]
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(model.lm_head, nn.Linear)
    parameters = model.model.layers[0].self_attn.q_proj.parameters()
    assert set(parameters) == {"weight", "scales", "biases"}


def test_checkpoint_config_is_atomic_and_validated(tmp_path) -> None:
    config = {
        "format": "mlx-minimax-music3-v1",
        "model": {},
        "components": {name: {} for name in (
            "language_model",
            "rvq_depth_decoder",
            "condition_encoder",
            "transformer",
            "vocoder",
        )},
        "quantization": dict(QUANTIZATION),
    }
    write_checkpoint_config(tmp_path, config)

    assert read_checkpoint_config(tmp_path) == config
    assert not (tmp_path / "config.json.tmp").exists()


def test_checkpoint_rejects_unknown_quantization(tmp_path) -> None:
    config = {
        "format": "mlx-minimax-music3-v1",
        "model": {},
        "components": {name: {} for name in (
            "language_model",
            "rvq_depth_decoder",
            "condition_encoder",
            "transformer",
            "vocoder",
        )},
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quantization"):
        read_checkpoint_config(tmp_path)


def test_sharded_component_weights_load_once(tmp_path) -> None:
    component = tmp_path / "language_model"
    component.mkdir()
    mx.save_safetensors(component / "part-a.safetensors", {"a": mx.ones((2,))})
    mx.save_safetensors(component / "part-b.safetensors", {"b": mx.zeros((3,))})
    index = {
        "metadata": {"total_size": 20},
        "weight_map": {"a": "part-a.safetensors", "b": "part-b.safetensors"},
    }
    (component / "model.safetensors.index.json").write_text(json.dumps(index))

    weights = load_component_weights(component)
    assert set(weights) == {"a", "b"}


def test_missing_indexed_shard_is_rejected(tmp_path) -> None:
    component = tmp_path / "transformer"
    component.mkdir()
    index = {"weight_map": {"a": "missing.safetensors"}}
    (component / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(FileNotFoundError, match="missing checkpoint shards"):
        load_component_weights(component)
