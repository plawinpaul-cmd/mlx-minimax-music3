from pathlib import Path

import pytest

from mlx_minimax_music3.cli import _text, build_parser


def test_generate_parser_accepts_file_inputs(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    lyrics = tmp_path / "lyrics.txt"
    prompt.write_text("warm pop")
    lyrics.write_text("hello")
    args = build_parser().parse_args(
        [
            "generate",
            "--prompt-file",
            str(prompt),
            "--lyrics-file",
            str(lyrics),
            "--output",
            str(tmp_path / "out.wav"),
        ]
    )
    assert args.command == "generate"
    assert args.duration == 60.0


def test_text_requires_one_nonempty_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _text(None, None, "prompt")
    empty = tmp_path / "empty.txt"
    empty.write_text("  ")
    with pytest.raises(ValueError, match="must not be empty"):
        _text(None, empty, "prompt")
