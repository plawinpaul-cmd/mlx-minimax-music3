import pytest

from mlx_minimax_music3.config import GenerationConfig, ModelConfig, OptimizationConfig


def test_default_model_contract_matches_official_pipeline() -> None:
    config = ModelConfig()

    assert config.frame_rate == 25
    assert config.num_codebooks == 8
    assert config.sampling_rate == 44_100
    assert config.max_audio_frames == 9_000


def test_duration_is_converted_to_frames_and_capped() -> None:
    model = ModelConfig()

    assert GenerationConfig(audio_duration=2.5).max_frames(model) == 62
    assert GenerationConfig(audio_duration=500).max_frames(model) == 9_000


@pytest.mark.parametrize("duration", [0, -1, 0.001])
def test_invalid_or_subframe_duration_is_rejected(duration: float) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(audio_duration=duration).max_frames(ModelConfig())


@pytest.mark.parametrize(
    "config",
    [GenerationConfig(num_inference_steps=0), GenerationConfig(top_k=0)],
)
def test_invalid_generation_parameters_are_rejected(config: GenerationConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_optimization_flags_can_be_disabled_independently() -> None:
    config = OptimizationConfig.from_env(
        {
            "MLX_MUSIC3_PRUNED_HEAD": "off",
            "MLX_MUSIC3_DEPTH_KV_CACHE": "0",
            "MLX_MUSIC3_BATCHED_DIT_CFG": "false",
            "MLX_MUSIC3_COMPILED_DIT": "no",
        }
    )

    assert config == OptimizationConfig(
        pruned_semantic_head=False,
        depth_kv_cache=False,
        batched_dit_cfg=False,
        compiled_dit=False,
    )


def test_invalid_optimization_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="MLX_MUSIC3_PRUNED_HEAD"):
        OptimizationConfig.from_env({"MLX_MUSIC3_PRUNED_HEAD": "sometimes"})
