from __future__ import annotations

import mlx.core as mx


def _finite_logits(logits: mx.array) -> mx.array:
    return mx.nan_to_num(
        logits.astype(mx.float32),
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )


def sample_top_k(
    logits: mx.array,
    key: mx.array,
    top_k: int = 50,
) -> tuple[mx.array, mx.array]:
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a non-empty vocabulary axis")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    values = _finite_logits(logits)
    k = min(top_k, values.shape[-1])
    threshold = mx.min(mx.topk(values, k, axis=-1), axis=-1, keepdims=True)
    filtered = mx.where(values < threshold, -mx.inf, values)
    next_key, sample_key = mx.random.split(key, 2)
    sample = mx.random.categorical(filtered, axis=-1, key=sample_key)
    return sample, next_key


def semantic_guided_logits(
    logits: mx.array,
    allowed_vocab: mx.array,
    cfg_scale: float = 1.5,
    conditional_top_k: int = 50,
) -> mx.array:
    if logits.ndim != 2 or logits.shape[0] != 2:
        raise ValueError("semantic logits must have shape [2, vocab]")
    if allowed_vocab.shape != (logits.shape[-1],):
        raise ValueError("allowed_vocab must match the vocabulary dimension")
    if conditional_top_k < 1:
        raise ValueError("conditional_top_k must be positive")

    values = _finite_logits(logits)
    conditional, unconditional = values[0:1], values[1:2]
    guided = unconditional + (conditional - unconditional) * cfg_scale
    k = min(conditional_top_k, conditional.shape[-1])
    threshold = mx.min(mx.topk(conditional, k, axis=-1), axis=-1, keepdims=True)
    keep = (conditional >= threshold) & allowed_vocab[None, :]
    return mx.where(keep, guided, -mx.inf)

