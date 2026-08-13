from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx

from .checkpoint import (
    CHECKPOINT_FORMAT,
    COMPONENTS,
    QUANTIZATION,
    is_quantized_path,
    read_checkpoint_config,
)
from .vocoder import (
    fold_weight_norm,
    pytorch_conv1d_to_mlx,
    pytorch_conv_transpose1d_to_mlx,
)

DEFAULT_SHARD_SIZE = 2 * 1024**3
MANIFEST_NAME = "conversion_manifest.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_weight_inventory(component_dir: Path) -> list[Path]:
    """Return the complete source shard inventory without requiring downloads."""
    for index_name in (
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ):
        index_path = component_dir / index_name
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid source shard index: {index_path}")
            return [component_dir / name for name in sorted(set(weight_map.values()))]
    else:
        files = [
            path
            for name in ("model.safetensors", "diffusion_pytorch_model.safetensors")
            if (path := component_dir / name).is_file()
        ]
    if not files:
        raise FileNotFoundError(f"no source safetensors found in {component_dir}")
    return files


def _source_weight_files(component_dir: Path) -> list[Path]:
    files = source_weight_inventory(component_dir)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source shards: {missing}")
    return files


def _as_bfloat16(value: mx.array) -> mx.array:
    if value.dtype in (mx.uint32, mx.int32, mx.int64):
        return value
    return value.astype(mx.bfloat16)


def convert_tensor(component: str, key: str, value: mx.array) -> dict[str, mx.array]:
    if key.endswith(".weight") and value.ndim == 2:
        module_path = key.removesuffix(".weight")
        if is_quantized_path(component, module_path):
            source = value.astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(source, **QUANTIZATION)
            return {
                key: packed,
                f"{module_path}.scales": scales,
                f"{module_path}.biases": biases,
            }

    if component == "condition_encoder" and key == "proj.weight":
        value = pytorch_conv1d_to_mlx(value)
    elif component == "transformer" and key in {
        "preprocess_conv.weight",
        "postprocess_conv.weight",
    }:
        value = pytorch_conv1d_to_mlx(value)
    return {key: _as_bfloat16(value)}


def _vocoder_weight_kind(module_path: str) -> str:
    return "transpose" if module_path.endswith("conv_t1") else "conv"


def convert_vocoder_weights(weights: Mapping[str, mx.array]) -> Iterator[tuple[str, mx.array]]:
    consumed: set[str] = set()
    for key in sorted(weights):
        if key in consumed:
            continue
        if key.endswith(".weight_g"):
            module_path = key.removesuffix(".weight_g")
            value_key = f"{module_path}.weight_v"
            if value_key not in weights:
                raise ValueError(f"missing paired weight normalization tensor: {value_key}")
            folded = fold_weight_norm(weights[key], weights[value_key], dim=0)
            if _vocoder_weight_kind(module_path) == "transpose":
                folded = pytorch_conv_transpose1d_to_mlx(folded)
            else:
                folded = pytorch_conv1d_to_mlx(folded)
            consumed.update((key, value_key))
            yield f"{module_path}.weight", _as_bfloat16(folded)
            continue
        if key.endswith(".weight_v"):
            if key not in consumed:
                raise ValueError(f"unpaired weight normalization tensor: {key}")
            continue
        value = weights[key]
        if key == "dec_in_proj.weight":
            value = pytorch_conv1d_to_mlx(value)
        elif key.endswith(".alpha"):
            if value.ndim != 3:
                raise ValueError(f"Snake alpha must have rank 3: {key}")
            value = value.transpose(0, 2, 1)
        yield key, _as_bfloat16(value)


def _converted_tensors(
    component: str,
    source_files: list[Path],
    completed: set[str],
) -> Iterator[tuple[str, mx.array]]:
    if component == "vocoder":
        if len(source_files) != 1:
            raise ValueError("vocoder conversion expects one source shard")
        weights = mx.load(source_files[0])
        for key, value in convert_vocoder_weights(weights):
            if key not in completed:
                yield key, value
        return

    for source_file in source_files:
        weights = mx.load(source_file)
        for source_key in sorted(weights):
            converted = convert_tensor(component, source_key, weights[source_key])
            for key, value in converted.items():
                if key not in completed:
                    yield key, value


def _new_manifest(component: str, source_files: list[Path]) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "component": component,
        "status": "in_progress",
        "quantization": dict(QUANTIZATION),
        "source_files": [path.name for path in source_files],
        "processed_source_files": [],
        "shards": [],
        "weight_map": {},
        "total_size": 0,
    }


def _load_resume_manifest(
    target_dir: Path,
    component: str,
    source_files: list[Path],
) -> dict[str, Any]:
    path = target_dir / MANIFEST_NAME
    if not path.exists():
        return _new_manifest(component, source_files)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = _new_manifest(component, source_files)
    for field in ("format", "component", "quantization", "source_files"):
        if manifest.get(field) != expected[field]:
            raise ValueError(f"resume manifest field {field!r} does not match this conversion")
    for shard in manifest.get("shards", []):
        shard_path = target_dir / shard["file"]
        if not shard_path.is_file():
            raise FileNotFoundError(f"resume shard is missing: {shard_path}")
        if shard_path.stat().st_size != shard["bytes"] or sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"resume shard failed integrity verification: {shard_path}")
    if "processed_source_files" not in manifest:
        manifest["processed_source_files"] = (
            list(manifest["source_files"]) if manifest.get("status") == "complete" else []
        )
    return manifest


def _next_shard_number(manifest: Mapping[str, Any]) -> int:
    numbers = []
    for shard in manifest.get("shards", []):
        match = re.fullmatch(r"model-(\d{5})\.safetensors", shard["file"])
        if not match:
            raise ValueError(f"unexpected output shard name: {shard['file']}")
        numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _flush_shard(
    target_dir: Path,
    tensors: dict[str, mx.array],
    manifest: dict[str, Any],
    number: int,
) -> None:
    if not tensors:
        return
    name = f"model-{number:05d}.safetensors"
    target = target_dir / name
    temporary = target.with_name(f"{target.stem}.tmp.safetensors")
    mx.save_safetensors(
        temporary,
        tensors,
        metadata={"format": CHECKPOINT_FORMAT, "component": manifest["component"]},
    )
    temporary.replace(target)
    size = target.stat().st_size
    tensor_sizes = {key: value.nbytes for key, value in tensors.items()}
    record = {
        "file": name,
        "bytes": size,
        "sha256": sha256_file(target),
        "tensors": sorted(tensors),
    }
    manifest["shards"].append(record)
    for key in tensors:
        manifest["weight_map"][key] = name
    manifest["total_size"] += sum(tensor_sizes.values())
    _atomic_json(target_dir / MANIFEST_NAME, manifest)
    mx.clear_cache()


def convert_component(
    source_root: Path,
    target_root: Path,
    component: str,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    if component not in {
        "language_model",
        "rvq_depth_decoder",
        "condition_encoder",
        "transformer",
        "vocoder",
    }:
        raise ValueError(f"unknown component: {component}")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")

    source_files = _source_weight_files(source_root / component)
    target_dir = target_root / component
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_resume_manifest(target_dir, component, source_files)
    if manifest.get("status") == "complete":
        return manifest

    completed = set(manifest["weight_map"])
    pending: dict[str, mx.array] = {}
    pending_size = 0
    shard_number = _next_shard_number(manifest)
    for key, value in _converted_tensors(component, source_files, completed):
        value_size = value.nbytes
        if pending and pending_size + value_size > shard_size:
            _flush_shard(target_dir, pending, manifest, shard_number)
            shard_number += 1
            pending = {}
            pending_size = 0
        pending[key] = value
        pending_size += value_size
    if pending:
        _flush_shard(target_dir, pending, manifest, shard_number)

    manifest["processed_source_files"] = sorted(manifest["source_files"])
    manifest["status"] = "complete"
    index = {
        "metadata": {"total_size": manifest["total_size"]},
        "weight_map": dict(sorted(manifest["weight_map"].items())),
    }
    _atomic_json(target_dir / "model.safetensors.index.json", index)
    _atomic_json(target_dir / MANIFEST_NAME, manifest)
    return manifest


def convert_component_source_shard(
    source_root: Path,
    target_root: Path,
    component: str,
    source_filename: str,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    """Convert one official source shard and leave a resumable component manifest."""
    if component not in {
        "language_model",
        "rvq_depth_decoder",
        "condition_encoder",
        "transformer",
        "vocoder",
    }:
        raise ValueError(f"unknown component: {component}")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if Path(source_filename).name != source_filename:
        raise ValueError("source_filename must be a basename")

    source_dir = source_root / component
    inventory = source_weight_inventory(source_dir)
    inventory_names = [path.name for path in inventory]
    if source_filename not in inventory_names:
        raise ValueError(f"source shard is not present in the official index: {source_filename}")
    source_file = source_dir / source_filename
    if not source_file.is_file():
        raise FileNotFoundError(f"source shard is not downloaded: {source_file}")

    target_dir = target_root / component
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_resume_manifest(target_dir, component, inventory)
    if source_filename in manifest["processed_source_files"]:
        return manifest
    if manifest.get("status") == "complete":
        raise ValueError("complete manifest does not record the requested source shard")

    completed = set(manifest["weight_map"])
    pending: dict[str, mx.array] = {}
    pending_size = 0
    shard_number = _next_shard_number(manifest)
    for key, value in _converted_tensors(component, [source_file], completed):
        value_size = value.nbytes
        if pending and pending_size + value_size > shard_size:
            _flush_shard(target_dir, pending, manifest, shard_number)
            shard_number += 1
            pending = {}
            pending_size = 0
        pending[key] = value
        pending_size += value_size
    if pending:
        _flush_shard(target_dir, pending, manifest, shard_number)

    manifest["processed_source_files"].append(source_filename)
    manifest["processed_source_files"].sort()
    if set(manifest["processed_source_files"]) == set(manifest["source_files"]):
        manifest["status"] = "complete"
        index = {
            "metadata": {"total_size": manifest["total_size"]},
            "weight_map": dict(sorted(manifest["weight_map"].items())),
        }
        _atomic_json(target_dir / "model.safetensors.index.json", index)
    _atomic_json(target_dir / MANIFEST_NAME, manifest)
    return manifest


def audit_checkpoint(model_path: Path, *, load_tensors: bool = True) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    read_checkpoint_config(model_path)
    if not (model_path / "tokenizer" / "tokenizer.json").is_file():
        raise FileNotFoundError("checkpoint tokenizer/tokenizer.json is missing")

    report: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "status": "pass",
        "components": {},
        "total_bytes": 0,
        "total_tensors": 0,
    }
    for component in COMPONENTS:
        component_dir = model_path / component
        manifest_path = component_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"component manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"component {component} is not complete")
        if manifest.get("component") != component or manifest.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"component manifest identity mismatch: {component}")

        index_path = component_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("weight_map") != dict(sorted(manifest["weight_map"].items())):
            raise ValueError(f"component index does not match manifest: {component}")

        component_bytes = 0
        component_tensors = 0
        for shard in manifest["shards"]:
            path = component_dir / shard["file"]
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint shard is missing: {path}")
            size = path.stat().st_size
            if size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
                raise ValueError(f"checkpoint shard failed integrity verification: {path}")
            if load_tensors:
                arrays = mx.load(path)
                if set(arrays) != set(shard["tensors"]):
                    raise ValueError(f"checkpoint shard tensor list mismatch: {path}")
                finite = [mx.all(mx.isfinite(value)) for value in arrays.values() if value.dtype != mx.uint32]
                if finite:
                    mx.eval(finite)
                    if not all(item.item() for item in finite):
                        raise ValueError(f"checkpoint shard contains non-finite tensors: {path}")
            component_bytes += size
            component_tensors += len(shard["tensors"])

        report["components"][component] = {
            "status": "pass",
            "bytes": component_bytes,
            "tensors": component_tensors,
            "shards": len(manifest["shards"]),
        }
        report["total_bytes"] += component_bytes
        report["total_tensors"] += component_tensors
    return report
