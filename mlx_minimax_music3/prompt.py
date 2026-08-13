from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from .config import ModelConfig

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CAPTION_START = "<|caption_start|>"
CAPTION_END = "<|caption_end|>"
LYRICS_START = "<|lyrics_start|>"
LYRICS_END = "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def clean_caption(caption: str) -> str:
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("prompt must be a non-empty string")

    def rewrite_special_tag(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("lyrics must be a non-empty string")

    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def build_prompt_text(prompt: str, lyrics: str) -> str:
    return (
        f"{IM_START}{CAPTION_START}{clean_caption(prompt)}{CAPTION_END}"
        f"{LYRICS_START}{normalize_lyrics(lyrics)}{LYRICS_END}{IM_END}{AUDIO_START}"
    )


def build_cfg_token_ids(
    prompt: str,
    lyrics: str,
    encode: Callable[[str], Sequence[int]],
    config: ModelConfig | None = None,
) -> np.ndarray:
    model = config or ModelConfig()
    conditional = np.asarray(encode(build_prompt_text(prompt, lyrics)), dtype=np.int32)
    if conditional.ndim != 1:
        raise ValueError(f"tokenizer must return one token sequence, got shape {conditional.shape}")
    if conditional.size > model.max_prompt_tokens:
        raise ValueError(
            f"assembled prompt has {conditional.size} tokens; maximum is {model.max_prompt_tokens}"
        )
    if conditional.size < 3:
        raise ValueError("assembled prompt must contain at least three tokens")
    unconditional = conditional.copy()
    unconditional[1:-2] = model.audio_cfg_token_id
    return np.stack((conditional, unconditional), axis=0)

