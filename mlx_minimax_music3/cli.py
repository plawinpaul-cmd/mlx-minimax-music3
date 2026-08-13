from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from .audio import write_wav
from .checkpoint import load_pipeline, resolve_model_path
from .config import GenerationConfig
from .conversion import audit_checkpoint

DEFAULT_MODEL = "vanch007/MiniMax-Music3-MLX-8bit"


def _text(value: str | None, file: Path | None, label: str) -> str:
    if (value is None) == (file is None):
        raise ValueError(f"provide exactly one of --{label} or --{label}-file")
    text = value if value is not None else file.expanduser().read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{label} must not be empty")
    return text


def _add_text_input(parser: argparse.ArgumentParser, name: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f"--{name}")
    group.add_argument(f"--{name}-file", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-minimax-music3")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate a stereo WAV")
    generate.add_argument("--model", default=DEFAULT_MODEL)
    _add_text_input(generate, "prompt")
    _add_text_input(generate, "lyrics")
    generate.add_argument("--duration", type=float, default=60.0)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--steps", type=int, default=30)
    generate.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", help="Verify checkpoint shards and manifests")
    verify.add_argument("--model", default=DEFAULT_MODEL)
    verify.add_argument("--metadata-only", action="store_true")

    serve = commands.add_parser("serve", help="Serve the official speech endpoint")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        prompt = _text(args.prompt, args.prompt_file, "prompt")
        lyrics = _text(args.lyrics, args.lyrics_file, "lyrics")
        config = GenerationConfig(
            audio_duration=args.duration,
            seed=args.seed,
            num_inference_steps=args.steps,
        )
        mx.metal.reset_peak_memory()
        started = time.perf_counter()
        pipeline = load_pipeline(args.model)
        result = pipeline.generate(prompt, lyrics, config)
        output = write_wav(args.output, result.audio, result.sampling_rate)
        elapsed = time.perf_counter() - started
        report = {
            "output": str(output),
            "sampling_rate": result.sampling_rate,
            "frames": result.num_frames,
            "chunks": result.num_chunks,
            "wall_seconds": elapsed,
            "peak_memory_gib": mx.metal.get_peak_memory() / 1024**3,
        }
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "verify":
        path = resolve_model_path(args.model)
        report = audit_checkpoint(path, load_tensors=not args.metadata_only)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(args.model), host=args.host, port=args.port)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
