import mlx.core as mx
import numpy as np

from mlx_minimax_music3.condition_encoder import ConditionEncoder, ConditionEncoderConfig
from mlx_minimax_music3.config import GenerationConfig, ModelConfig
from mlx_minimax_music3.flow_transformer import FlowTransformer, FlowTransformerConfig
from mlx_minimax_music3.language_model import LanguageModel, LanguageModelConfig
from mlx_minimax_music3.pipeline import (
    MiniMaxMusic3Pipeline,
    PipelineComponents,
    _flow_cfg_velocity,
)
from mlx_minimax_music3.rvq_decoder import RVQDecoderConfig, RVQDepthDecoder
from mlx_minimax_music3.vocoder import Vocoder, VocoderConfig


class TinyTokenizer:
    def encode(self, text: str):
        return [1, 2, 3, 4, 5]


def tiny_pipeline() -> MiniMaxMusic3Pipeline:
    model_config = ModelConfig(
        frame_rate=25,
        num_codebooks=4,
        semantic_vocab_size=8,
        audio_vocab_size=16,
        audio_code_offset=20,
        audio_end_token_id=10,
        audio_cfg_token_id=11,
        max_prompt_tokens=20,
        max_audio_frames=20,
        sampling_rate=44_100,
        latent_hop_length=4,
        latent_channels=8,
    )
    language = LanguageModel(
        LanguageModelConfig(
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=64,
            max_position_embeddings=32,
            rope_theta=10_000,
        )
    )
    # Make the semantic AR choice deterministic: all hidden states remain positive
    # and audio token 20 is the unique maximum.
    language.model.embed_tokens.weight = mx.ones((64, 8))
    layer = language.model.layers[0]
    for linear in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ):
        linear.weight = mx.zeros_like(linear.weight)
    head = np.zeros((64, 8), dtype=np.float32)
    head[20] = 1.0
    language.lm_head.weight = mx.array(head)

    decoder = RVQDepthDecoder(
        RVQDecoderConfig(
            hidden_size=8,
            num_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            audio_vocab_size=16,
            num_codebooks=4,
            max_position_embeddings=8,
        )
    )
    condition = ConditionEncoder(
        ConditionEncoderConfig(
            condition_hidden_dim=8,
            num_condition_layers=4,
            out_dim=6,
            input_sampling_rate=1,
            input_hop_length=1,
            output_sampling_rate=2,
            output_hop_length=1,
        )
    )
    transformer = FlowTransformer(
        FlowTransformerConfig(
            in_channels=8,
            condition_dim=6,
            num_layers=1,
            num_attention_heads=2,
            attention_head_dim=4,
            ff_inner_dim=12,
            rotary_dim=2,
            fourier_embedding_dim=4,
        )
    )
    vocoder = Vocoder(
        VocoderConfig(
            latent_channels=8,
            decoder_input_dim=16,
            decoder_hidden_dim=32,
            upsampling_ratios=(2, 2),
            sampling_rate=44_100,
        )
    )
    return MiniMaxMusic3Pipeline(
        PipelineComponents(
            tokenizer=TinyTokenizer(),
            language_model=language,
            rvq_depth_decoder=decoder,
            condition_encoder=condition,
            transformer=transformer,
            vocoder=vocoder,
        ),
        model_config=model_config,
    )


def test_tiny_pipeline_runs_end_to_end_on_mlx() -> None:
    mx.random.seed(23)
    pipeline = tiny_pipeline()
    result = pipeline.generate(
        "warm synth pop",
        "[verse]\nhello",
        GenerationConfig(audio_duration=0.08, seed=7, num_inference_steps=2, top_k=1),
    )
    mx.eval(result.audio)

    assert result.num_frames == 2
    assert result.num_chunks == 1
    assert result.sampling_rate == 44_100
    assert result.audio.shape == (1, 2, 16)
    assert np.isfinite(np.asarray(result.audio)).all()
    assert np.abs(np.asarray(result.audio)).max() <= 1.0


def test_pipeline_is_deterministic_for_fixed_seed() -> None:
    mx.random.seed(29)
    pipeline = tiny_pipeline()
    config = GenerationConfig(audio_duration=0.08, seed=42, num_inference_steps=2, top_k=1)
    first = pipeline.generate("warm synth pop", "hello", config).audio
    second = pipeline.generate("warm synth pop", "hello", config).audio
    mx.eval(first, second)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_prompt_cfg_pair_is_created() -> None:
    ids = tiny_pipeline().encode_prompt("warm synth pop", "hello")
    mx.eval(ids)

    np.testing.assert_array_equal(np.asarray(ids[0]), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.asarray(ids[1]), [1, 11, 11, 4, 5])


def test_batched_flow_cfg_matches_separate_forwards() -> None:
    mx.random.seed(43)
    pipeline = tiny_pipeline()
    latents = mx.random.normal((1, 7, 8))
    condition = mx.random.normal((1, 7, 6))
    timestep = mx.array([0.5])

    separate = _flow_cfg_velocity(
        pipeline.components.transformer,
        latents,
        timestep,
        condition,
        1.7,
        batched=False,
    )
    batched = _flow_cfg_velocity(
        pipeline.components.transformer,
        latents,
        timestep,
        condition,
        1.7,
        batched=True,
    )
    mx.eval(separate, batched)

    np.testing.assert_allclose(
        np.asarray(batched),
        np.asarray(separate),
        atol=1e-5,
        rtol=1e-5,
    )


def test_compiled_flow_cfg_matches_eager_forward() -> None:
    mx.random.seed(47)
    pipeline = tiny_pipeline()
    latents = mx.random.normal((1, 7, 8))
    condition = mx.random.normal((1, 7, 6))
    timestep = mx.array([0.5])

    eager = _flow_cfg_velocity(
        pipeline.components.transformer,
        latents,
        timestep,
        condition,
        1.7,
        batched=True,
    )
    compiled = _flow_cfg_velocity(
        mx.compile(pipeline.components.transformer),
        latents,
        timestep,
        condition,
        1.7,
        batched=True,
    )
    mx.eval(eager, compiled)

    np.testing.assert_allclose(
        np.asarray(compiled),
        np.asarray(eager),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dit_compile_is_limited_to_profitable_shapes() -> None:
    pipeline = tiny_pipeline()

    assert pipeline._flow_transformer(689, 30) is pipeline.components.transformer
    assert pipeline._flow_transformer(172, 5) is pipeline.components.transformer
    assert pipeline._flow_transformer(310, 30) is not pipeline.components.transformer
