#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_minimax_music3.audio import write_wav
from mlx_minimax_music3.checkpoint import load_pipeline
from mlx_minimax_music3.config import GenerationConfig, OptimizationConfig


MODES = {
    "baseline": OptimizationConfig(
        pruned_semantic_head=False,
        depth_kv_cache=False,
        batched_dit_cfg=False,
        compiled_dit=False,
    ),
    "pruned-head": OptimizationConfig(
        pruned_semantic_head=True,
        depth_kv_cache=False,
        batched_dit_cfg=False,
        compiled_dit=False,
    ),
    "head-depth-kv": OptimizationConfig(
        pruned_semantic_head=True,
        depth_kv_cache=True,
        batched_dit_cfg=False,
        compiled_dit=False,
    ),
    "optimized": OptimizationConfig(),
}

DEFAULT_PROMPT = (
    "C-Pop melodic ballad, 68 BPM, Eb major, male tenor, warm piano, "
    "strings, restrained verses and an expansive emotional chorus."
)
DEFAULT_LYRICS = "[verse]\n花开南去后，东海几年别。\n[chorus]\n相思深夜后，秋色共谁寻。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Music3 MLX optimization stages")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--lyrics", default=DEFAULT_LYRICS)
    parser.add_argument("--audio-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    optimization = MODES[args.mode]
    generation = GenerationConfig(
        audio_duration=args.duration,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    mx.reset_peak_memory()
    total_started = time.perf_counter()

    started = time.perf_counter()
    pipeline = load_pipeline(args.model, optimization_config=optimization)
    load_seconds = time.perf_counter() - started

    text_ids = pipeline.encode_prompt(args.prompt, args.lyrics)
    key = mx.random.key(generation.seed)

    started = time.perf_counter()
    frame_hiddens, key = pipeline.generate_frame_hiddens(text_ids, generation, key)
    ar_seconds = time.perf_counter() - started

    started = time.perf_counter()
    latent_chunks, key = pipeline.denoise_chunks(frame_hiddens, generation, key)
    dit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    audio = pipeline.decode_chunks(latent_chunks)
    mx.eval(audio, key)
    vocoder_seconds = time.perf_counter() - started

    output = None
    if args.audio_output is not None:
        output = str(write_wav(args.audio_output, audio, pipeline.model_config.sampling_rate))

    values = np.asarray(audio)
    report = {
        "status": "pass",
        "mode": args.mode,
        "optimization": asdict(optimization),
        "model": str(args.model.expanduser().resolve()),
        "hardware": platform.machine(),
        "macos": platform.mac_ver()[0],
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "load_average": list(os.getloadavg()),
        "duration_seconds": args.duration,
        "inference_steps": args.steps,
        "seed": args.seed,
        "prompt_tokens": int(text_ids.shape[1]),
        "frames": int(frame_hiddens.shape[1]),
        "chunks": len(latent_chunks),
        "timing_seconds": {
            "load": load_seconds,
            "autoregressive": ar_seconds,
            "dit": dit_seconds,
            "vocoder": vocoder_seconds,
            "generation": ar_seconds + dit_seconds + vocoder_seconds,
            "total": time.perf_counter() - total_started,
        },
        "peak_memory_gib": mx.get_peak_memory() / 1024**3,
        "active_memory_gib": mx.get_active_memory() / 1024**3,
        "audio": {
            "output": output,
            "shape": list(values.shape),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "rms": float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
