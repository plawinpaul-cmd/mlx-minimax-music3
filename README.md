# mlx-minimax-music3

Native Apple MLX inference for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

> [!IMPORTANT]
> This port is under active development. No converted checkpoint or verified end-to-end generation is available yet.

The approved architecture and verification requirements are recorded in
[`docs/spark/2026-08-14-minimax-music3-mlx-design.md`](docs/spark/2026-08-14-minimax-music3-mlx-design.md).

## Intended Runtime

- Apple silicon with Metal
- Python 3.11-3.13
- MLX 0.31.x and MLX-LM 0.31.x
- Native MLX model execution without a PyTorch inference fallback

## Status

| Deliverable | Status |
|---|---|
| Native MLX components | pending |
| Mixed 8-bit/BF16 conversion | pending |
| Local music generation | pending |
| GitHub release | pending |
| Hugging Face checkpoint | pending |

## License

This derivative project is distributed under the MiniMax-Music3 Community License. See [LICENSE](LICENSE). MiniMax Music 3 incorporates work derived from Qwen3, Stable Audio, and Descript Audio Codec; their upstream notices remain applicable as described in the license.

