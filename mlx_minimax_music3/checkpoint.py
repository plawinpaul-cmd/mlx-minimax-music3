from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .condition_encoder import ConditionEncoder, ConditionEncoderConfig
from .config import ModelConfig
from .flow_transformer import FlowTransformer, FlowTransformerConfig
from .language_model import LanguageModel, LanguageModelConfig
from .pipeline import MiniMaxMusic3Pipeline, PipelineComponents
from .rvq_decoder import RVQDecoderConfig, RVQDepthDecoder
from .vocoder import Vocoder, VocoderConfig

CHECKPOINT_FORMAT = "mlx-minimax-music3-v1"
QUANTIZATION = {"group_size": 64, "bits": 8, "mode": "affine"}
COMPONENTS = (
    "language_model",
    "rvq_depth_decoder",
    "condition_encoder",
    "transformer",
    "vocoder",
)


def is_quantized_path(component: str, path: str) -> bool:
    if component == "language_model":
        return path.startswith("model.layers.") and any(
            marker in path
            for marker in (
                ".self_attn.q_proj",
                ".self_attn.k_proj",
                ".self_attn.v_proj",
                ".self_attn.o_proj",
                ".mlp.gate_proj",
                ".mlp.up_proj",
                ".mlp.down_proj",
            )
        )
    if component == "rvq_depth_decoder":
        return path.startswith("layers.") and any(
            marker in path
            for marker in (
                ".attn.to_q",
                ".attn.to_k",
                ".attn.to_v",
                ".attn.to_out",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
            )
        )
    if component == "transformer":
        if path in {"proj_in", "proj_out"}:
            return True
        return path.startswith("transformer_blocks.") and any(
            marker in path
            for marker in (
                ".attn.to_q",
                ".attn.to_k",
                ".attn.to_v",
                ".attn.to_out.0",
                ".ff_in",
                ".ff_out",
            )
        )
    return False


def validate_quantization(quantization: Mapping[str, object]) -> dict[str, int | str]:
    if set(quantization) != {"group_size", "bits", "mode"}:
        raise ValueError("checkpoint quantization metadata has unsupported fields")
    group_size = quantization.get("group_size")
    bits = quantization.get("bits")
    mode = quantization.get("mode")
    if group_size != 64 or bits not in {4, 8} or mode != "affine":
        raise ValueError("checkpoint quantization metadata is not supported")
    return {"group_size": int(group_size), "bits": int(bits), "mode": str(mode)}


def quantization_for_component(config: Mapping[str, object], component: str) -> dict[str, int | str]:
    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    default = config.get("quantization")
    if not isinstance(default, dict):
        raise ValueError("checkpoint quantization metadata is missing")
    overrides = config.get("component_quantization", {})
    if not isinstance(overrides, dict):
        raise ValueError("checkpoint component quantization metadata is invalid")
    override = overrides.get(component, default)
    if not isinstance(override, dict):
        raise ValueError(f"checkpoint quantization metadata is invalid for {component}")
    return validate_quantization(override)


def apply_component_quantization(
    component: str,
    model: nn.Module,
    quantization: Mapping[str, object] = QUANTIZATION,
) -> list[str]:
    quantization = validate_quantization(quantization)
    quantized: list[str] = []

    def predicate(path: str, module: nn.Module) -> bool:
        selected = (
            isinstance(module, nn.Linear)
            and module.weight.shape[-1] % quantization["group_size"] == 0
            and is_quantized_path(component, path)
        )
        if selected:
            quantized.append(path)
        return selected

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization["mode"],
        class_predicate=predicate,
    )
    return quantized


def component_config_dict(
    model_config: ModelConfig,
    language_model: LanguageModelConfig,
    rvq_depth_decoder: RVQDecoderConfig,
    condition_encoder: ConditionEncoderConfig,
    transformer: FlowTransformerConfig,
    vocoder: VocoderConfig,
    *,
    quantization: Mapping[str, object] = QUANTIZATION,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "model": asdict(model_config),
        "components": {
            "language_model": asdict(language_model),
            "rvq_depth_decoder": asdict(rvq_depth_decoder),
            "condition_encoder": asdict(condition_encoder),
            "transformer": asdict(transformer),
            "vocoder": asdict(vocoder),
        },
        "quantization": validate_quantization(quantization),
    }


def write_checkpoint_config(path: Path, config: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    target = path / "config.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def read_checkpoint_config(path: Path) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {config.get('format')!r}")
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("checkpoint quantization metadata is missing")
    validate_quantization(quantization)
    component_quantization = config.get("component_quantization", {})
    if not isinstance(component_quantization, dict):
        raise ValueError("checkpoint component quantization metadata is invalid")
    unknown = set(component_quantization) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"checkpoint has quantization for unknown components: {sorted(unknown)}")
    for component in component_quantization:
        quantization_for_component(config, component)
    missing = set(COMPONENTS) - set(config.get("components", {}))
    if missing:
        raise ValueError(f"checkpoint is missing component configs: {sorted(missing)}")
    return config


def _weight_files(component_dir: Path) -> list[Path]:
    index_path = component_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid shard index: {index_path}")
        names = sorted(set(weight_map.values()))
        files = [component_dir / name for name in names]
    else:
        single = component_dir / "model.safetensors"
        files = [single] if single.exists() else []
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint shards: {missing}")
    if not files:
        raise FileNotFoundError(f"no checkpoint weights found in {component_dir}")
    return files


def load_component_weights(component_dir: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for file in _weight_files(component_dir):
        shard = mx.load(file)
        overlap = set(weights) & set(shard)
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)}")
        weights.update(shard)
    return weights


def load_component(
    component: str,
    model: nn.Module,
    model_path: Path,
    quantization: Mapping[str, object] = QUANTIZATION,
) -> nn.Module:
    apply_component_quantization(component, model, quantization)
    weights = load_component_weights(model_path / component)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def resolve_model_path(model: str | Path) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        return local.resolve()
    return Path(snapshot_download(str(model)))


def load_pipeline(model: str | Path) -> MiniMaxMusic3Pipeline:
    model_path = resolve_model_path(model)
    config = read_checkpoint_config(model_path)
    components = config["components"]

    language = load_component(
        "language_model",
        LanguageModel(LanguageModelConfig.from_dict(components["language_model"])),
        model_path,
        quantization_for_component(config, "language_model"),
    )
    rvq = load_component(
        "rvq_depth_decoder",
        RVQDepthDecoder(RVQDecoderConfig(**components["rvq_depth_decoder"])),
        model_path,
        quantization_for_component(config, "rvq_depth_decoder"),
    )
    condition = load_component(
        "condition_encoder",
        ConditionEncoder(ConditionEncoderConfig(**components["condition_encoder"])),
        model_path,
        quantization_for_component(config, "condition_encoder"),
    )
    transformer = load_component(
        "transformer",
        FlowTransformer(FlowTransformerConfig(**components["transformer"])),
        model_path,
        quantization_for_component(config, "transformer"),
    )
    vocoder = load_component(
        "vocoder",
        Vocoder(VocoderConfig(**components["vocoder"])),
        model_path,
        quantization_for_component(config, "vocoder"),
    )

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer" / "tokenizer.json"))
    return MiniMaxMusic3Pipeline(
        PipelineComponents(
            tokenizer=tokenizer,
            language_model=language,
            rvq_depth_decoder=rvq,
            condition_encoder=condition,
            transformer=transformer,
            vocoder=vocoder,
        ),
        model_config=ModelConfig(**config["model"]),
    )
