# mlx-minimax-music3

Native Apple MLX inference for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3), including a selective 8-bit/BF16 checkpoint, local CLI, Python API, and HTTP service.

- Code: [vanch007/mlx-minimax-music3](https://github.com/vanch007/mlx-minimax-music3)
- Model: [vanch007/MiniMax-Music3-MLX-8bit](https://huggingface.co/vanch007/MiniMax-Music3-MLX-8bit)
- Source model revision: `c2509fd6b60d1ae169cd1df27f78a53174ba17e8`
- Diffusers reference revision: `c6da9936e4bda83107943a16eb8682e9a37d8527`

The production path uses MLX on Metal. It does not import or execute PyTorch.

## Requirements

- Apple silicon Mac with Metal support
- Python 3.11, 3.12, or 3.13
- About 13.2 GiB for model weights
- Sufficient unified-memory headroom

The release was validated on an Apple M3 Max with 128 GB unified memory, macOS 27.0, Python 3.12.13, MLX 0.31.2, and MLX-LM 0.31.3. A 10-second, two-window, 30-step generation peaked at 23.34 GiB of MLX memory. Machines with 32 GB may be tight once macOS and other applications are included; this release has not been validated on lower-memory Macs.

## Install

With `uv`:

```bash
git clone https://github.com/vanch007/mlx-minimax-music3.git
cd mlx-minimax-music3
uv sync --extra server
```

Or install the tagged release into an existing environment:

```bash
python -m pip install "mlx-minimax-music3[server] @ git+https://github.com/vanch007/mlx-minimax-music3.git@v0.1.0"
```

## Generate Music

The default model is downloaded from Hugging Face on first use:

```bash
uv run mlx-minimax-music3 generate \
  --prompt "Genre: acoustic pop. BPM: 96. Warm female lead, fingerpicked guitar, soft piano, brushed drums." \
  --lyrics $'[verse]\nMorning light across the sea\n[chorus]\nStay here and sing this song with me' \
  --duration 10 \
  --seed 7 \
  --steps 30 \
  --output song.wav
```

Use an already downloaded local checkpoint by passing its directory:

```bash
uv run mlx-minimax-music3 generate \
  --model models/MiniMax-Music3-MLX-8bit \
  --prompt-file prompt.txt \
  --lyrics-file lyrics.txt \
  --duration 10 \
  --output song.wav
```

Python API:

```python
from mlx_minimax_music3 import generate, load_model

model = load_model("vanch007/MiniMax-Music3-MLX-8bit")
audio, sample_rate = generate(
    model,
    prompt="Warm acoustic pop with intimate female vocals.",
    lyrics="[verse]\nMorning light...",
    audio_duration=10.0,
    seed=7,
    num_inference_steps=30,
)
```

## Local Service

```bash
uv run mlx-minimax-music3 serve \
  --model models/MiniMax-Music3-MLX-8bit \
  --host 127.0.0.1 \
  --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Generate WAV through the OpenAI-style speech route:

```bash
curl -o song.wav http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimax_ttm",
    "input": "[verse]\nMorning light across the sea",
    "instructions": "warm acoustic pop with female vocals",
    "max_new_tokens": 250,
    "seed": 7,
    "stream": false
  }'
```

`max_new_tokens` is the maximum number of 25 Hz audio frames. Streaming output is not supported by this release.

## Verify A Checkpoint

```bash
uv run mlx-minimax-music3 verify \
  --model models/MiniMax-Music3-MLX-8bit
```

The verifier checks the checkpoint format, component manifests, shard sizes, SHA-256 hashes, tensor names, and finite values.

## Checkpoint Format

Affine 8-bit quantization uses group size 64 for large linear matrices in:

- Qwen3 attention and MLP layers
- RVQ depth-decoder attention and MLP layers
- Flow-transformer projections, attention, and feed-forward layers

Embeddings, logits heads, normalizations, convolutions, timestep projections, the condition encoder, and the complete vocoder remain BF16. The converted weights contain 1,978 MLX tensors across 12 safetensors shards and occupy 14,167,660,156 bytes.

## Validation

The retained [component parity report](reports/component-parity.json) compares the official BF16 PyTorch implementation with this quantized MLX release on fixed inputs:

| Component | Cosine similarity | Relative RMSE |
|---|---:|---:|
| Language model | 0.999822 | 0.025001 |
| RVQ depth decoder | 0.999916 | 0.012956 |
| Condition encoder | 0.999990 | 0.004654 |
| Flow transformer | 0.998990 | 0.045071 |
| Vocoder | 0.999877 | 0.015715 |

Local release validation also passed:

- 91 automated tests
- Strict load and finite-value audit of all 12 shards
- 10-second generation with 250 AR frames, two overlapping chunks, and 30 flow steps
- 44.1 kHz, 16-bit PCM, stereo output lasting 9.996 seconds
- No clipped samples; measured peak 0.4582
- Real HTTP `/health` and `/v1/audio/speech` calls

Automated checks do not establish perceptual quality equivalence. Human listening review is recorded as `pending`.

## Limitations

- Apple silicon only
- Single generated song per request
- Non-streaming WAV output
- No training, fine-tuning, LoRA, or batch-generation support
- Generation can stop earlier than the requested maximum duration
- Prompt adherence and lyric pronunciation inherit limitations of the source model

## Development

```bash
uv sync --extra server --extra test
uv run pytest
```

The approved architecture and evidence requirements are in [the design specification](docs/spark/2026-08-14-minimax-music3-mlx-design.md).

## License

This project and the converted weights are distributed under the [MiniMax-Music3 Community License](LICENSE). Review its attribution, acceptable-use, safeguard, and commercial terms before deployment.
