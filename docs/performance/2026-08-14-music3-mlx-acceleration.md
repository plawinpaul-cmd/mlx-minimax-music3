# MiniMax Music 3 MLX Acceleration Research

Date: 2026-08-14

## Scope

This note inventories public native-MLX implementations available on the release day and ranks acceleration work for the local M3 Max port. It separates measured evidence from estimates.

## Public implementations

### ddalcu/mlx-serve

- Native Zig + MLX engine with a published 8-bit, group-64 Music 3 pack.
- Engine and model are ready to run through MLX Core.app or an HTTP endpoint.
- The project reports 29.4 ms per AR frame on an M-series 128 GB machine after optimization, down from 43.8 ms.
- Its measured AR changes are directly relevant to this port:
  - depth-decoder KV cache: 23.3 to 11.7 ms per frame;
  - semantic-head pruning: 2.6 to 0.5 ms per frame;
  - total AR stage: 43.8 to 29.4 ms per frame.
- The engine is MIT licensed; model weights retain the MiniMax Music 3 license.

Sources:

- https://github.com/ddalcu/mlx-serve
- https://github.com/ddalcu/mlx-serve/blob/main/docs/gotchas/models-media.md
- https://huggingface.co/ddalcu/MiniMax-Music3-MLX-Serve-8bit

### pierre427/mlx-minimax-music3

- Python MLX port using mlx-lm's Qwen3 backbone; no converted model repository is published.
- The documented path requires the approximately 57 GB upstream checkpoint.
- On M5 Max, the author reports a 6-second test falling from 27.8 to 11.0 seconds with M5 TF32 plus q6 AR quantization.
- The implementation also demonstrates a single batch-2 DiT forward for CFG and optional semantic-head slicing.
- M5 TF32/NAX results do not transfer to M3 Max. Q6 AR, batch-2 CFG, and head slicing remain relevant.

Source: https://github.com/pierre427/mlx-minimax-music3

### Search conclusion

At the time of research, GitHub exposed this repository and Pierre's as dedicated Python MLX ports. Hugging Face additionally exposed ddalcu's MLX-Serve model pack. No MiniMax Music 3 implementation was present in mlx-audio or mlx-examples. This is a public-index result, not proof that no private or unindexed port exists.

## Local bottlenecks visible in source

1. The 200,000-row language-model head is evaluated every audio frame, then masked to 16,385 legal outputs. The head is also left dense BF16 by the current quantization predicate.
2. The four-layer depth decoder re-forwards the full growing sequence seven times for every frame; it has no KV cache.
3. Every flow step calls the 36-layer DiT twice, once for conditional and once for unconditional CFG, instead of one batch-2 call.
4. DiT RoPE tables and zero conditions are rebuilt on repeated calls.
5. The AR loop forces more than one evaluation barrier per frame and retains a Python list of frame graphs.
6. The server has no prompt-KV reuse for repeated seeds or variations of the same long caption and lyrics.

## Ranked plan

### P0: exact or near-exact changes

1. Prune the semantic head at load or conversion time to the 16,384 audio-code rows plus the end token. Quantize the retained head and depth projection/audio heads. This is row-selection exact and should remove about 1.5 GB from the current dense BF16 head before quantization.
2. Add an exact per-frame KV cache to the depth decoder. Reset it for each frame and feed only the newly appended codebook token after the initial two-token prefill.
3. Run DiT CFG as one batch-2 forward per Euler step, then combine the two rows. Cache RoPE tables by sequence length and create the zero condition once per chunk.
4. Collapse avoidable AR synchronization to the sampled-token read, and periodically materialize a preallocated frame buffer rather than synchronizing every hidden-state step.

### P1: request-level acceleration

5. Add a small LRU prompt cache keyed by normalized caption, lyrics, tokenizer revision, and model revision. This does not improve the first request, but it removes long Qwen prefill for new seeds using the same input.
6. Benchmark fixed-shape `mx.compile` only after P0. MLX documents graph fusion and lower memory use, but shape changes trigger recompilation, so compiling the whole dynamic pipeline first is not justified.

### P2: quality-gated alternatives

7. Build a q6 AR variant and compare fixed-trajectory code ranks, component parity, and listening samples against q8. Do not make q4 depth the default: mlx-serve measured only a small total speed gain with lower replay agreement.
8. A/B test ddalcu's released engine locally before investing in a large Python rewrite. Its model pack is about 13 GB, so storage must be checked before download.

## Do not prioritize

- M5-only TF32/NAX acceleration on M3 Max.
- Quantized KV cache for this short Music 3 context. Current mlx-lm reports show it can increase peak memory and reduce throughput.
- Global `mx.compile` without a stage benchmark.
- Prompt truncation as a silent default; it changes the conditioning contract and needs an explicit fast mode plus listening validation.

## Benchmark validity

The official-demo run produced three valid 44.1 kHz stereo WAV files, but its latency data is not a release baseline. It crossed battery and AC power states, host load varied sharply, and the controller was interrupted before aggregation. The JSON report is retained with `status: invalid_environment` so the timing is not accidentally promoted to a performance claim.

## Recommended next experiment

Implement P0 behind feature flags, add stage timers for prompt prefill, global AR decode, depth decode, DiT, and vocoder, then compare the same 250-frame official case on AC power. Require output parity for head pruning and depth caching, and report stage-level deltas rather than only end-to-end wall time.

## Implementation outcome

P0 was implemented behind independent environment and Python API switches:

| Change | Controlled local evidence | Default |
|---|---|---|
| Semantic-head pruning | All 16,385 retained logits exact; top-50 set exact; active MLX memory 13.268 to 11.866 GiB | on |
| Depth KV cache | Single-frame median 113.1 to 54.5 ms; top-1 RVQ codes equal; mean hidden-state absolute difference 0.00239 | on |
| Batch-2 DiT CFG | Same-model median 2.810 to 2.663 s; mean output absolute difference 0.00000155 | on |
| RoPE cache | Shape-keyed eager arrays reused | on |
| Fixed-shape compile | Helpful at 310 positions, harmful at 689; enabled only at <=384 positions and >=12 steps | adaptive |

The adjacent 10-second high-load runs reduced observed generation time from 155.29 to 117.62 seconds and AR time from 46.53 to 23.43 seconds before adaptive compile was added. These numbers are retained only as provisional evidence because host load changed between runs. A clean-host end-to-end baseline remains `pending`; controlled component A/B results are the acceptance evidence for the implementation.

The optimized and compatibility paths need not produce identical fixed-seed waveforms. Head row selection itself is exact, but categorical sampling over a compact vocabulary changes random-number mapping, and cached attention introduces small BF16 rounding differences. The quality/listening comparison therefore remains `pending` rather than being described as perceptually identical.

Additional primary references:

- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://github.com/ml-explore/mlx-lm#long-prompts-and-generations
- https://github.com/ml-explore/mlx-lm/issues/1587
