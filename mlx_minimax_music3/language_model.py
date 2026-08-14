from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3


@dataclass(frozen=True)
class LanguageModelConfig:
    model_type: str = "qwen3"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12_288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 200_000
    max_position_embeddings: int = 10_240
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LanguageModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    def mlx_lm_args(self) -> qwen3.ModelArgs:
        return qwen3.ModelArgs(
            model_type=self.model_type,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
        )


class LanguageModel(nn.Module):
    """Qwen3 backbone exposing hidden states required by Music3 generation."""

    def __init__(self, config: LanguageModelConfig | None = None):
        super().__init__()
        self.config = config or LanguageModelConfig()
        args = self.config.mlx_lm_args()
        self.model = qwen3.Qwen3Model(args)
        self.semantic_head: nn.Linear | None = None
        self._semantic_head_layout: tuple[int, int, int] | None = None
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def prepare_semantic_head(
        self,
        audio_code_offset: int,
        semantic_vocab_size: int,
        audio_end_token_id: int,
    ) -> None:
        """Retain only Music3 semantic-code and end-token output rows."""
        stop = audio_code_offset + semantic_vocab_size
        layout = (audio_code_offset, semantic_vocab_size, audio_end_token_id)
        if semantic_vocab_size < 1 or audio_code_offset < 0 or stop > self.config.vocab_size:
            raise ValueError("semantic vocabulary is outside the language-model vocabulary")
        if audio_end_token_id < 0 or audio_end_token_id >= self.config.vocab_size:
            raise ValueError("audio end token is outside the language-model vocabulary")
        if audio_code_offset <= audio_end_token_id < stop:
            raise ValueError("audio end token must be outside the semantic-code range")
        if self._semantic_head_layout is not None:
            if self._semantic_head_layout != layout:
                raise ValueError("semantic head was already prepared with a different layout")
            return

        if self.config.tie_word_embeddings:
            source_weight = self.model.embed_tokens.weight
        else:
            source_weight = self.lm_head.weight
        semantic_weight = mx.concatenate(
            (
                source_weight[audio_code_offset:stop],
                source_weight[audio_end_token_id : audio_end_token_id + 1],
            ),
            axis=0,
        )
        head = nn.Linear(
            self.config.hidden_size,
            semantic_vocab_size + 1,
            bias=False,
        )
        head.weight = semantic_weight
        mx.eval(head.weight)
        self.semantic_head = head
        self._semantic_head_layout = layout
        if not self.config.tie_word_embeddings:
            self.lm_head = None
        mx.clear_cache()

    def hidden_states(
        self,
        input_ids: mx.array | None = None,
        *,
        input_embeddings: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of input_ids or input_embeddings")
        if input_ids is None:
            input_ids = mx.zeros((input_embeddings.shape[0], 0), dtype=mx.int32)
        return self.model(input_ids, cache=cache, input_embeddings=input_embeddings)

    def logits(self, hidden_states: mx.array) -> mx.array:
        if self.semantic_head is not None:
            return self.semantic_head(hidden_states)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        input_embeddings: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        return self.logits(
            self.hidden_states(input_ids, input_embeddings=input_embeddings, cache=cache)
        )

    @property
    def layers(self):
        return self.model.layers

    @property
    def semantic_head_layout(self) -> tuple[int, int, int] | None:
        return self._semantic_head_layout
