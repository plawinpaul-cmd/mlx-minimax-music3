#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


COMPONENTS = (
    "language_model",
    "rvq_depth_decoder",
    "condition_encoder",
    "transformer",
    "vocoder",
)
DEFAULT_SOURCE_REPO = "MiniMaxAI/MiniMax-Music3"
DEFAULT_SOURCE_REVISION = "c2509fd6b60d1ae169cd1df27f78a53174ba17e8"
DEFAULT_REFERENCE_REVISION = "c6da9936e4bda83107943a16eb8682e9a37d8527"

# The BF16 components should remain very close across frameworks. Quantized
# components include both 8-bit weight error and backend numerical differences.
THRESHOLDS = {
    "condition_encoder": {"min_cosine": 0.999, "max_relative_rmse": 0.05},
    "vocoder": {"min_cosine": 0.995, "max_relative_rmse": 0.15},
    "rvq_depth_decoder": {"min_cosine": 0.99, "max_relative_rmse": 0.20},
    "transformer": {"min_cosine": 0.98, "max_relative_rmse": 0.30},
    "language_model": {"min_cosine": 0.98, "max_relative_rmse": 0.30},
}

SOURCE_LAYOUT = {
    "language_model": ("model.safetensors.index.json", None),
    "transformer": ("diffusion_pytorch_model.safetensors.index.json", None),
    "rvq_depth_decoder": (None, "diffusion_pytorch_model.safetensors"),
    "condition_encoder": (None, "diffusion_pytorch_model.safetensors"),
    "vocoder": (None, "diffusion_pytorch_model.safetensors"),
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _component_config(model_path: Path, component: str) -> dict:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    return config["components"][component]


def make_inputs(component: str, path: Path, seed: int = 7) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    if component == "language_model":
        arrays = {"input_ids": np.array([[1, 220, 151643, 151674]], dtype=np.int64)}
    elif component == "rvq_depth_decoder":
        arrays = {"inputs_embeds": (rng.standard_normal((2, 3, 4096)) * 0.1).astype(np.float32)}
    elif component == "condition_encoder":
        arrays = {"hidden_states": (rng.standard_normal((1, 3, 8 * 4096)) * 0.1).astype(np.float32)}
    elif component == "transformer":
        arrays = {
            "hidden_states": (rng.standard_normal((1, 5, 128)) * 0.1).astype(np.float32),
            "encoder_hidden_states": (rng.standard_normal((1, 5, 2048)) * 0.1).astype(np.float32),
            "timestep": np.array([0.5], dtype=np.float32),
        }
    elif component == "vocoder":
        arrays = {"latents": (rng.standard_normal((1, 4, 128)) * 0.1).astype(np.float32)}
    else:
        raise ValueError(f"unknown component: {component}")
    np.savez(path, **arrays)
    return {name: list(value.shape) for name, value in arrays.items()}


def parity_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool | list[int]]:
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, actual={actual.shape}")
    reference = reference.astype(np.float64, copy=False)
    actual = actual.astype(np.float64, copy=False)
    delta = actual - reference
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    rmse = float(np.sqrt(np.mean(delta**2)))
    denominator = float(np.linalg.norm(reference.ravel()) * np.linalg.norm(actual.ravel()))
    cosine = float(np.dot(reference.ravel(), actual.ravel()) / denominator) if denominator else 1.0
    return {
        "shape": list(reference.shape),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": rmse / max(reference_rms, 1e-12),
        "cosine_similarity": cosine,
    }


def metric_status(component: str, metrics: dict) -> str:
    threshold = THRESHOLDS[component]
    passed = (
        metrics["finite"]
        and metrics["cosine_similarity"] >= threshold["min_cosine"]
        and metrics["relative_rmse"] <= threshold["max_relative_rmse"]
    )
    return "pass" if passed else "fail"


def _reference_class(component: str):
    if component == "language_model":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        return Qwen3ForCausalLM, Qwen3Config

    from diffusers import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    classes = {
        "rvq_depth_decoder": MiniMaxMusic3RVQDepthDecoder,
        "condition_encoder": MiniMaxMusic3ConditionEncoder,
        "transformer": MiniMaxMusic3Transformer1DModel,
        "vocoder": MiniMaxMusic3Vocoder,
    }
    return classes[component], None


def _build_reference_model(component: str, config: dict):
    import torch

    model_class, config_class = _reference_class(component)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        if config_class is not None:
            model = model_class(config_class(**config))
        else:
            parameters = inspect.signature(model_class.__init__).parameters
            model = model_class(**{key: value for key, value in config.items() if key in parameters})
    finally:
        torch.set_default_dtype(old_dtype)
    return model


def _source_shards(
    component: str,
    repo_id: str,
    revision: str,
    local_dir: Path,
) -> list[str]:
    from huggingface_hub import hf_hub_download

    index_name, single_name = SOURCE_LAYOUT[component]
    if single_name is not None:
        return [single_name]
    index_path = Path(
        hf_hub_download(
            repo_id,
            filename=f"{component}/{index_name}",
            revision=revision,
            local_dir=local_dir,
        )
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return sorted(set(index["weight_map"].values()))


def _load_reference_weights(
    model,
    component: str,
    repo_id: str,
    revision: str,
    local_dir: Path,
) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    expected = set(model.state_dict())
    loaded: set[str] = set()
    shard_names = _source_shards(component, repo_id, revision, local_dir)
    for name in shard_names:
        source_path = Path(
            hf_hub_download(
                repo_id,
                filename=f"{component}/{name}",
                revision=revision,
                local_dir=local_dir,
            )
        )
        weights = load_file(source_path, device="cpu")
        overlap = loaded & set(weights)
        if overlap:
            raise ValueError(f"duplicate source tensors: {sorted(overlap)}")
        result = model.load_state_dict(weights, strict=False, assign=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected source tensors: {result.unexpected_keys}")
        loaded.update(weights)
        del weights
        gc.collect()
        source_path.unlink()
    missing = expected - loaded
    unexpected = loaded - expected
    if missing or unexpected:
        raise ValueError(
            f"reference state coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {"source_shards": shard_names, "source_tensors": len(loaded)}


def _reference_forward(component: str, model, inputs: dict[str, np.ndarray], device: str) -> np.ndarray:
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available")
    model = model.eval().to(device)
    with torch.inference_mode():
        if component == "language_model":
            output = model(input_ids=torch.from_numpy(inputs["input_ids"]).to(device)).logits
        elif component == "rvq_depth_decoder":
            value = torch.from_numpy(inputs["inputs_embeds"]).to(device=device, dtype=torch.bfloat16)
            output = model(value)
        elif component == "condition_encoder":
            value = torch.from_numpy(inputs["hidden_states"]).to(device=device, dtype=torch.bfloat16)
            output = model(value)
        elif component == "transformer":
            hidden = torch.from_numpy(inputs["hidden_states"]).to(device=device, dtype=torch.bfloat16)
            condition = torch.from_numpy(inputs["encoder_hidden_states"]).to(
                device=device, dtype=torch.bfloat16
            )
            timestep = torch.from_numpy(inputs["timestep"]).to(device=device, dtype=torch.bfloat16)
            output = model(
                hidden_states=hidden.transpose(1, 2),
                timestep=timestep,
                encoder_hidden_states=condition,
                return_dict=False,
            )[0].transpose(1, 2)
        elif component == "vocoder":
            value = torch.from_numpy(inputs["latents"]).to(device=device, dtype=torch.bfloat16)
            output = model(value.transpose(1, 2))
        else:
            raise ValueError(f"unknown component: {component}")
    return output.float().cpu().numpy()


def reference_worker(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    model_path = args.model.expanduser().resolve()
    inputs = dict(np.load(args.input))
    config = _component_config(model_path, args.component)
    model = _build_reference_model(args.component, config)
    with tempfile.TemporaryDirectory(prefix=f"minimax-ref-{args.component}-") as temporary:
        coverage = _load_reference_weights(
            model,
            args.component,
            args.source_repo,
            args.source_revision,
            Path(temporary),
        )
    output = _reference_forward(args.component, model, inputs, args.reference_device)
    np.save(args.output, output)
    return {
        "backend": "pytorch-reference",
        "device": args.reference_device,
        "seconds": time.perf_counter() - started,
        **coverage,
    }


def mlx_worker(args: argparse.Namespace) -> dict:
    import mlx.core as mx

    from mlx_minimax_music3.checkpoint import load_component, read_checkpoint_config
    from mlx_minimax_music3.condition_encoder import ConditionEncoder, ConditionEncoderConfig
    from mlx_minimax_music3.flow_transformer import FlowTransformer, FlowTransformerConfig
    from mlx_minimax_music3.language_model import LanguageModel, LanguageModelConfig
    from mlx_minimax_music3.rvq_decoder import RVQDecoderConfig, RVQDepthDecoder
    from mlx_minimax_music3.vocoder import Vocoder, VocoderConfig

    started = time.perf_counter()
    model_path = args.model.expanduser().resolve()
    checkpoint = read_checkpoint_config(model_path)
    config = checkpoint["components"][args.component]
    constructors = {
        "language_model": lambda: LanguageModel(LanguageModelConfig.from_dict(config)),
        "rvq_depth_decoder": lambda: RVQDepthDecoder(RVQDecoderConfig(**config)),
        "condition_encoder": lambda: ConditionEncoder(ConditionEncoderConfig(**config)),
        "transformer": lambda: FlowTransformer(FlowTransformerConfig(**config)),
        "vocoder": lambda: Vocoder(VocoderConfig(**config)),
    }
    model = load_component(args.component, constructors[args.component](), model_path)
    inputs = dict(np.load(args.input))
    if args.component == "language_model":
        output = model(mx.array(inputs["input_ids"], dtype=mx.int32))
    elif args.component == "rvq_depth_decoder":
        output = model(mx.array(inputs["inputs_embeds"], dtype=mx.bfloat16))
    elif args.component == "condition_encoder":
        output = model(mx.array(inputs["hidden_states"], dtype=mx.bfloat16))
    elif args.component == "transformer":
        output = model(
            mx.array(inputs["hidden_states"], dtype=mx.bfloat16),
            mx.array(inputs["timestep"], dtype=mx.bfloat16),
            mx.array(inputs["encoder_hidden_states"], dtype=mx.bfloat16),
        )
    elif args.component == "vocoder":
        output = model(mx.array(inputs["latents"], dtype=mx.bfloat16))
    else:
        raise ValueError(f"unknown component: {args.component}")
    output = output.astype(mx.float32)
    mx.eval(output)
    np.save(args.output, np.asarray(output))
    return {
        "backend": "mlx",
        "device": "metal",
        "seconds": time.perf_counter() - started,
        "peak_memory_gib": mx.get_peak_memory() / 2**30,
    }


def _run_worker(python: Path, script: Path, args: list[str], env: dict[str, str]) -> dict:
    completed = subprocess.run(
        [str(python), str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode:
        raise RuntimeError(
            f"worker failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def compare(args: argparse.Namespace) -> dict:
    model_path = args.model.expanduser().resolve()
    reference_source = args.reference_source.expanduser().resolve()
    # Keep the virtual-environment entry point intact. Resolving its symlink
    # bypasses the environment's site-packages and invokes the base interpreter.
    reference_python = Path(os.path.abspath(args.reference_python.expanduser()))
    actual_revision = subprocess.check_output(
        ["git", "-C", str(reference_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != args.reference_revision:
        raise ValueError(
            f"reference source revision mismatch: expected {args.reference_revision}, got {actual_revision}"
        )

    selected = COMPONENTS if args.component == "all" else (args.component,)
    report = {
        "format": "mlx-minimax-music3-component-parity-v1",
        "status": "in_progress",
        "checkpoint": getattr(args, "checkpoint_id", None) or str(model_path),
        "checkpoint_revision": args.checkpoint_revision,
        "source_model": args.source_repo,
        "source_revision": args.source_revision,
        "reference_implementation": "huggingface/diffusers",
        "reference_revision": args.reference_revision,
        "reference_precision": "BF16",
        "release_precision": "selective affine 8-bit group-size 64 with BF16 exceptions",
        "components": {},
    }
    if args.report.is_file():
        existing = json.loads(args.report.read_text(encoding="utf-8"))
        identity = (
            "checkpoint_revision",
            "source_revision",
            "reference_revision",
        )
        if all(existing.get(key) == report[key] for key in identity):
            report["components"].update(existing.get("components", {}))

    script = Path(__file__).resolve()
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(reference_source / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    for index, component in enumerate(selected):
        with tempfile.TemporaryDirectory(prefix=f"minimax-parity-{component}-") as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "inputs.npz"
            reference_path = temporary_path / "reference.npy"
            mlx_path = temporary_path / "mlx.npy"
            input_shapes = make_inputs(component, input_path, seed=args.seed + index)
            common = [
                "--model",
                str(model_path),
                "--component",
                component,
                "--input",
                str(input_path),
            ]
            reference_metadata = _run_worker(
                reference_python,
                script,
                [
                    "--worker",
                    "reference",
                    *common,
                    "--output",
                    str(reference_path),
                    "--source-repo",
                    args.source_repo,
                    "--source-revision",
                    args.source_revision,
                    "--reference-device",
                    args.reference_device,
                ],
                env,
            )
            mlx_metadata = _run_worker(
                Path(sys.executable),
                script,
                ["--worker", "mlx", *common, "--output", str(mlx_path)],
                os.environ.copy(),
            )
            metrics = parity_metrics(np.load(reference_path), np.load(mlx_path))
            status = metric_status(component, metrics)
            report["components"][component] = {
                "status": status,
                "input_shapes": input_shapes,
                "thresholds": THRESHOLDS[component],
                "metrics": metrics,
                "reference": reference_metadata,
                "mlx": mlx_metadata,
            }
            statuses = {item["status"] for item in report["components"].values()}
            if "fail" in statuses:
                report["status"] = "fail"
            elif set(report["components"]) == set(COMPONENTS):
                report["status"] = "pass"
            else:
                report["status"] = "in_progress"
            _atomic_json(args.report, report)
            print(json.dumps({"component": component, **report["components"][component]}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare official PyTorch and native MLX Music3 components")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", choices=(*COMPONENTS, "all"), default="all")
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--reference-revision", default=DEFAULT_REFERENCE_REVISION)
    parser.add_argument("--checkpoint-revision", required=False, default="local")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--reference-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--report", type=Path, default=Path("reports/component-parity.json"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--worker", choices=("compare", "reference", "mlx"), default="compare")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "reference":
        print(json.dumps(reference_worker(args)))
        return 0
    if args.worker == "mlx":
        print(json.dumps(mlx_worker(args)))
        return 0
    if args.reference_python is None or args.reference_source is None:
        raise ValueError("--reference-python and --reference-source are required for comparison")
    report = compare(args)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
