#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_minimax_music3.checkpoint import COMPONENTS
from mlx_minimax_music3.conversion import convert_component
from mlx_minimax_music3.prepare import prepare_checkpoint_layout


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MiniMax Music 3 Diffusers weights to MLX")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=(*COMPONENTS, "all"),
        default="all",
    )
    parser.add_argument("--shard-size-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.shard_size_gib <= 0:
        parser.error("--shard-size-gib must be positive")

    prepare_checkpoint_layout(args.source, args.output)
    selected = COMPONENTS if args.component == "all" else (args.component,)
    reports = {}
    for component in selected:
        reports[component] = convert_component(
            args.source,
            args.output,
            component,
            shard_size=int(args.shard_size_gib * 1024**3),
        )
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

