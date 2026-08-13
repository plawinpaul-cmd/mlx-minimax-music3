from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .checkpoint import component_config_dict, write_checkpoint_config
from .condition_encoder import ConditionEncoderConfig
from .config import ModelConfig
from .flow_transformer import FlowTransformerConfig
from .language_model import LanguageModelConfig
from .rvq_decoder import RVQDecoderConfig
from .vocoder import VocoderConfig


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"source config is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_config_from_source(source_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    language_source = _json(source_root / "language_model" / "config.json")
    rope = language_source.get("rope_parameters", {})
    language_values = dict(language_source)
    if "rope_theta" not in language_values and "rope_theta" in rope:
        language_values["rope_theta"] = rope["rope_theta"]

    language = LanguageModelConfig.from_dict(language_values)
    rvq = RVQDecoderConfig(
        **{
            key: value
            for key, value in _json(source_root / "rvq_depth_decoder" / "config.json").items()
            if key in RVQDecoderConfig.__dataclass_fields__
        }
    )
    condition = ConditionEncoderConfig(
        **{
            key: value
            for key, value in _json(source_root / "condition_encoder" / "config.json").items()
            if key in ConditionEncoderConfig.__dataclass_fields__
        }
    )
    transformer = FlowTransformerConfig(
        **{
            key: value
            for key, value in _json(source_root / "transformer" / "config.json").items()
            if key in FlowTransformerConfig.__dataclass_fields__
        }
    )
    vocoder_values = {
        key: value
        for key, value in _json(source_root / "vocoder" / "config.json").items()
        if key in VocoderConfig.__dataclass_fields__
    }
    if "upsampling_ratios" in vocoder_values:
        vocoder_values["upsampling_ratios"] = tuple(vocoder_values["upsampling_ratios"])
    vocoder = VocoderConfig(**vocoder_values)

    model = ModelConfig(
        num_codebooks=rvq.num_codebooks,
        audio_vocab_size=rvq.audio_vocab_size,
        sampling_rate=vocoder.sampling_rate,
        latent_hop_length=vocoder.hop_length,
        latent_channels=vocoder.latent_channels,
    )
    if condition.num_condition_layers != rvq.num_codebooks:
        raise ValueError("source condition and RVQ codebook counts disagree")
    if condition.condition_hidden_dim != language.hidden_size:
        raise ValueError("source condition and language hidden sizes disagree")
    if condition.out_dim != transformer.condition_dim:
        raise ValueError("source condition and transformer dimensions disagree")
    if transformer.in_channels != vocoder.latent_channels:
        raise ValueError("source transformer and vocoder latent channels disagree")

    return component_config_dict(model, language, rvq, condition, transformer, vocoder)


def prepare_checkpoint_layout(source_root: Path, target_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    config = checkpoint_config_from_source(source_root)
    write_checkpoint_config(target_root, config)

    tokenizer_source = source_root / "tokenizer"
    tokenizer_target = target_root / "tokenizer"
    tokenizer_target.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = tokenizer_source / name
        if source.is_file():
            shutil.copy2(source, tokenizer_target / name)
    if not (tokenizer_target / "tokenizer.json").is_file():
        raise FileNotFoundError("source tokenizer/tokenizer.json is missing")

    license_source = source_root / "LICENSE"
    if not license_source.is_file():
        raise FileNotFoundError("source LICENSE is missing")
    shutil.copy2(license_source, target_root / "LICENSE")
    return config

