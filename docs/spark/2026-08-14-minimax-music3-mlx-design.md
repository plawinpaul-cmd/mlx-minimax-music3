# MiniMax Music 3 Native MLX Port Design

- Date: 2026-08-14
- Status: approved, implemented, and locally validated; public release audit pending
- Owner: vanch007
- Source code: https://github.com/MiniMax-AI/MiniMax-Music3
- Source model: https://huggingface.co/MiniMaxAI/MiniMax-Music3
- Reference implementation: https://github.com/huggingface/diffusers/pull/14456
- Target code repository: `vanch007/mlx-minimax-music3`
- Target model repository: `vanch007/MiniMax-Music3-MLX-8bit`

## 1. Objective

Create and verify a native Apple MLX implementation of MiniMax Music 3 for local Apple Silicon inference. Convert the official model into an MLX-oriented mixed 8-bit/BF16 checkpoint, deploy it on the local M3 Max with 128 GB unified memory, and publish the reproducible code and converted model to the owner's GitHub and Hugging Face accounts.

Completion requires all of the following:

1. The autoregressive language model, RVQ depth decoder, condition encoder, flow-matching transformer, and vocoder execute with MLX on Metal. PyTorch is not part of the production inference path.
2. Official weights are converted reproducibly into documented MLX safetensors with a selective 8-bit/BF16 policy.
3. A local command generates a real, non-silent, 44.1 kHz stereo WAV from lyrics and a music description.
4. A Python API, CLI, and OpenAI-compatible `/v1/audio/speech` endpoint are available.
5. Component parity, end-to-end behavior, resource use, and public remote artifacts are verified with retained evidence.

## 2. Evidence And Constraints

The design is based on the official checkpoint and the pending Hugging Face Diffusers integration at commit `c6da9936e4bda83107943a16eb8682e9a37d8527`.

- Architecture: Qwen3-based 8B global autoregressive model, 0.6B local RVQ depth decoder, 2.4B flow transformer, condition encoder, and DAC-style waveform decoder.
- Output contract: native 44.1 kHz stereo waveform from the Diffusers pipeline. The older SGLang server downsamples to 32 kHz; this port preserves the newer native output.
- Frame contract: 25 autoregressive frames per second, up to 9,000 frames.
- Chunk contract: 200-frame denoising windows, 100-frame hop, overlap carry and cropping matching the Diffusers reference.
- Flow contract: Euler flow matching, 30 steps by default, CFG scale 1.7.
- Local runtime evidence: MLX 0.31.0 exposes Metal execution, `Conv1d`, `ConvTranspose1d`, `RMSNorm`, `LayerNorm`, quantized linear layers, and fast scaled dot-product attention.
- Storage constraint: the full Hugging Face repository is about 53.4 GiB because it contains legacy and Diffusers layouts. The needed Diffusers component set is about 26.6 GiB. The internal disk cannot safely retain both source and converted copies at once.
- License: redistribution and modification are allowed by the MiniMax-Music3 Community License subject to its notice, acceptable-use, attribution, and commercial terms. Derived repositories must preserve the full license and source attribution.

## 3. Chosen Approach

Use a fully native MLX runtime with selective affine 8-bit quantization, group size 64, and BF16 for numerically or perceptually sensitive parameters.

This approach was selected over pure BF16 because it materially reduces local storage and runtime memory while retaining higher precision for embeddings, logits, normalization, convolutions, timestep projections, and waveform synthesis. A simultaneous BF16 release is out of scope for the first completed release because it would exceed the current safe conversion workspace and double validation work.

No PyTorch fallback will be hidden behind the default command. PyTorch may exist only in an isolated development parity environment used to compare the official reference implementation with MLX.

## 4. Repository Architecture

The code repository will use focused modules with explicit ownership:

```text
mlx-minimax-music3/
  mlx_minimax_music3/
    config.py              checkpoint and generation configuration
    loading.py             model resolution, validation, and lazy loading
    sampling.py            deterministic top-k sampling and RNG policy
    prompt.py              official prompt cleanup and token contract
    language_model.py      Qwen3 global AR model and KV cache
    rvq_decoder.py         seven residual codebooks per audio frame
    condition_encoder.py   hidden-state mixing, projection, resampling
    flow_transformer.py    1D flow-matching transformer
    scheduler.py           reference Euler schedule and overlap blending
    vocoder.py             native MLX DAC-style stereo decoder
    pipeline.py            orchestration and component lifecycle
    audio.py               WAV normalization and writing
    server.py              `/v1/audio/speech` service
    cli.py                 local command-line interface
  scripts/
    convert.py             resumable selective conversion entry point
    verify_checkpoint.py   manifest, shape, dtype, and checksum audit
    compare_components.py  PyTorch/MLX development parity runner
    benchmark.py           runtime and memory evidence
  tests/
  docs/spark/
  reports/
  pyproject.toml
  README.md
  LICENSE
```

The model repository will retain the loading contract rather than duplicate runtime source:

```text
MiniMax-Music3-MLX-8bit/
  config.json
  quantization.json
  model.safetensors.index.json
  model-*.safetensors
  tokenizer/
  conversion_manifest.json
  source_manifest.json
  LICENSE
  README.md
```

## 5. Native MLX Components

### 5.1 Prompt And Tokenizer

The port will reproduce the reference caption cleanup, lyric normalization, special-token sequence, conditional/unconditional prompt pair, token limits, and audio token IDs exactly. Hugging Face `tokenizers` may run on CPU; this is preprocessing rather than model inference.

### 5.2 Global Language Model

The Qwen3 model will be implemented with MLX layers and grouped-query attention. It will keep a growing KV cache and process only one feedback token per generated frame after prompt prefill. The two CFG rows remain resident together. Semantic sampling will preserve the official vocabulary mask, conditional top-50 restriction, CFG scale 1.5, and top-50 sampling.

### 5.3 RVQ Depth Decoder

For every semantic frame, the local decoder will autoregressively generate residual codebooks 1 through 7. It will retain causal attention across the short depth sequence, positional embeddings, seven output heads, and the continuous hidden states needed by the synthesis path.

### 5.4 Condition Encoder

Eight hidden-state groups will be mixed with learned softmax weights, scaled, convolved, and nearest-neighbor resampled from the 24 kHz/960-hop timeline to the 44.1 kHz/512-hop latent timeline. Layout conversion between PyTorch channel-first tensors and MLX channel-last convolutions will be isolated and tested.

### 5.5 Flow Transformer And Scheduler

The transformer will preserve the learned random Fourier timestep projection, partial rotary embedding, 36 attention blocks, gated feed-forward layers, residual 1x1 convolutions, and conditional/unconditional passes. Long songs will use the official window, overlap prompt, latent carry, and crop constants. Chunk results will be released as soon as later processing no longer needs them.

### 5.6 Vocoder

The vocoder will use native MLX convolutions and transposed convolutions. PyTorch weight normalization pairs will be folded during conversion as `weight = g * v / ||v||`, so inference stores ordinary convolution kernels. Latent channels will be folded into two decoder rows and reshaped back to stereo. Snake activations, dilation, padding, upsampling ratios, cropping, clamping, and tanh output will match the reference.

## 6. Quantization Policy

Default format: affine 8-bit, group size 64, with BF16 exceptions.

Quantize:

- Qwen3 attention projections and MLP matrices.
- RVQ decoder attention projections and MLP matrices.
- Flow transformer attention projections, input/output projections, and feed-forward matrices when the input dimension is group-compatible.

Keep BF16:

- All normalization parameters and biases.
- Token, residual-codebook, and positional embeddings.
- Language-model and RVQ output heads.
- Condition encoder parameters.
- Fourier and timestep embeddings.
- All convolution and transposed-convolution weights.
- Entire vocoder.

Every tensor's final dtype and quantization decision will be recorded in `conversion_manifest.json`. The loader will reject a checkpoint whose manifest, expected keys, shapes, or quantization metadata do not agree.

## 7. Conversion And Storage Flow

Conversion must be resumable, content-addressed, and bounded by local disk space.

1. Fetch only metadata, tokenizer files, licenses, and the selected Diffusers component.
2. Convert one source shard or component at a time.
3. Transpose convolution kernels and split or map projections according to an explicit key plan.
4. Quantize eligible matrices and write an output shard no larger than 2 GiB.
5. Reopen the output shard, verify keys, shapes, finite values, and checksums.
6. Upload the verified shard to a staging branch in the target Hugging Face repository.
7. Confirm the remote object size and checksum metadata before deleting only the temporary source files downloaded by this task.
8. Resume safely from the manifest after interruption.
9. Commit the final model card and manifests only after every component is present.

Deletion is limited to task-owned temporary source files. Existing caches, user models, and unrelated files must never be removed.

## 8. Runtime Interfaces

### Python

```python
from mlx_minimax_music3 import load_model, generate

model = load_model("vanch007/MiniMax-Music3-MLX-8bit")
audio, sample_rate = generate(
    model,
    prompt="Warm acoustic pop with intimate female vocals.",
    lyrics="[verse]\nMorning light...",
    audio_duration=10.0,
    seed=7,
)
```

### CLI

```bash
mlx-minimax-music3 generate \
  --prompt-file prompt.txt \
  --lyrics-file lyrics.txt \
  --duration 10 \
  --seed 7 \
  --output output.wav
```

### Local Service

```bash
mlx-minimax-music3 serve --host 127.0.0.1 --port 8000
```

The service will accept the official request fields `model`, `input`, `instructions`, `response_format`, `seed`, `max_new_tokens`, and `stream`. Only `stream=false` and WAV output are supported initially, matching the model's current non-streaming limitation. Unsupported values return a clear 4xx response.

## 9. Memory And Lifecycle

The M3 Max target has enough unified memory for resident 8-bit AR and synthesis components, but the runtime will still use explicit lifecycle boundaries:

- Load AR components, generate frame hidden states, and clear their caches.
- Load or activate synthesis components for flow matching and vocoding.
- Decode and crop one chunk at a time where possible.
- Call MLX evaluation at controlled boundaries and clear Metal cache between major stages when measurements show a benefit.

The default path prioritizes correct native execution and reproducibility. Compilation and more aggressive residency optimizations are accepted only after parity is retained.

## 10. Error Handling

The loader and interfaces will fail early for:

- Unsupported hardware, missing Metal support, or incompatible MLX versions.
- Missing model shards, checksum mismatch, wrong tensor shape, or incompatible quantization metadata.
- Empty prompt or lyrics, prompt length over 5,000 tokens, non-positive duration, or requests over 9,000 frames.
- Immediate end-of-audio with zero frames.
- NaN or infinite values at component boundaries.
- Disk exhaustion during conversion.
- Failed or incomplete remote upload verification.

An interrupted conversion remains `in_progress`; it must never be presented or tagged as a complete model release.

## 11. Verification Plan

### Static And Unit Evidence

- Prompt construction and token IDs match the reference fixture.
- Weight key plans cover every source tensor exactly once or identify an intentional omission.
- Kernel transpositions and folded weight normalization match PyTorch equations.
- Causal masks, rotary embedding, KV cache, sampling masks, schedules, chunk boundaries, overlap carry, and crop lengths have focused tests.
- Checkpoint audit proves every required tensor, shape, dtype, and shard checksum.

### Component Parity

Development-only PyTorch reference tests will use fixed small inputs and compare BF16 MLX component outputs before quantization. Each component report will record maximum absolute error, mean absolute error, cosine similarity, and tolerances justified by the component precision. The 8-bit release will be compared separately and labeled as quantized rather than bitwise equivalent.

### End-To-End Evidence

At minimum, retain:

1. A short deterministic smoke generation that completes without PyTorch imports in the production process.
2. A vocal sample with lyrics and a structured music description.
3. WAV validation proving 44,100 Hz, two channels, finite samples, non-silence, expected duration range, and no severe clipping.
4. Runtime report with model size, wall time, real-time factor, peak resident memory or Metal footprint, and hardware/software versions.
5. Listening-review status recorded separately as `pass`, `fail`, or `pending`; automated metrics do not substitute for listening.

No claim of quality parity with the official model will be made unless corresponding listening and reference evidence exists.

### Publication Evidence

- GitHub repository is public and its default branch contains tagged, tested code and documentation.
- Hugging Face model repository is public and all referenced shards resolve with a fresh-cache checkpoint audit.
- A fresh temporary environment installs the GitHub release, downloads the public model, and performs a smoke generation.
- READMEs link to each other, identify the source model and license, and show commands actually exercised locally.

## 12. Release Sequence

1. Implement and test model modules using synthetic fixtures.
2. Create the private or draft Hugging Face target and stream converted shards to it.
3. Complete component parity and checkpoint verification.
4. Run short end-to-end local generation and resource measurement.
5. Finish documentation, license notices, known limitations, and model card.
6. Push the code repository and tag the first verified release.
7. Make or finalize the model repository and run a fresh-cache public smoke test.
8. Publish verification reports and only then describe the port as complete.

Creating repositories and uploading model data are authorized by the approved objective. Destructive actions remain limited by the conversion policy above; rewriting Git history, force-pushing, deleting remote repositories, or removing unrelated local data are not authorized.

## 13. Out Of Scope For The First Release

- Training, fine-tuning, LoRA, or optimizer support.
- Streaming audio generation.
- Batch generation beyond the conditional/unconditional pair.
- iOS application or graphical web interface.
- A second BF16 or 4-bit public checkpoint.
- Strict symbolic guarantees for tempo, key, instrumentation, lyrics, or song structure beyond the source model's capabilities.

These exclusions do not narrow the requested local MLX deployment, conversion, adaptation, or publication deliverables.

## 14. Acceptance Matrix

| Requirement | Required evidence | Current status |
|---|---|---|
| Native MLX project | Production dependency/runtime audit and successful Metal generation | pass |
| Converted MLX model | Complete manifest, shard audit, and public HF resolution | local and private-remote audit pass; public resolution pending |
| Local deployment | Reproducible local command and valid generated WAV | pass |
| MLX adaptation | Component parity plus pipeline and API tests | pass |
| GitHub publication | Public repository, tag, and remote file inspection | pending |
| Hugging Face publication | Public model repository and fresh-cache generation | pending |

The project is complete only when every row is supported by current retained evidence. A passing narrow unit test or successful upload alone is insufficient.
