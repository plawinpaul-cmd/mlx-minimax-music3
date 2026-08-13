import numpy as np
import pytest

from mlx_minimax_music3.config import ModelConfig
from mlx_minimax_music3.prompt import (
    build_cfg_token_ids,
    build_prompt_text,
    clean_caption,
    normalize_lyrics,
)


def test_caption_cleanup_matches_reference_rules() -> None:
    caption = """## Global Metadata
- **Genre:** *folk pop*

---
• Warm mix
<|bpm 96|>"""

    assert clean_caption(caption) == "Global Metadata\nGenre: folk pop\nWarm mix\nbpm is 96"


def test_lyrics_keep_leading_tags_and_drop_same_line_text() -> None:
    lyrics = "[Verse] This text is dropped\nHello [CHORUS]\n[Bridge] [Solo] ignored"

    assert normalize_lyrics(lyrics) == (
        "[start]\n[verse]\nHello\n[chorus]\n[bridge]\n[solo]"
    )


def test_prompt_wrapper_is_byte_exact() -> None:
    assert build_prompt_text("Warm pop", "[Verse]\nHello") == (
        "<|im_start|><|caption_start|>Warm pop<|caption_end|>"
        "<|lyrics_start|>[start]\n[verse]\nHello<|lyrics_end|>"
        "<|im_end|><|audio_start|>"
    )


def test_cfg_ids_preserve_edges_and_mask_interior() -> None:
    ids = build_cfg_token_ids("Warm pop", "Hello", lambda _: [1, 2, 3, 4, 5])

    np.testing.assert_array_equal(ids[0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(ids[1], [1, 151654, 151654, 4, 5])


def test_prompt_token_limit_is_enforced() -> None:
    config = ModelConfig(max_prompt_tokens=4)
    with pytest.raises(ValueError, match="maximum is 4"):
        build_cfg_token_ids("Warm pop", "Hello", lambda _: [1, 2, 3, 4, 5], config)


@pytest.mark.parametrize("field", ["", "  ", None])
def test_empty_inputs_are_rejected(field) -> None:
    with pytest.raises(ValueError):
        build_prompt_text(field, "lyrics")
    with pytest.raises(ValueError):
        build_prompt_text("prompt", field)
