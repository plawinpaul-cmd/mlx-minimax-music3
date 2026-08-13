import json

from mlx_minimax_music3.prepare import (
    checkpoint_config_from_source,
    prepare_checkpoint_layout,
)


def write_source_configs(root) -> None:
    configs = {
        "language_model": {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 200000,
            "max_position_embeddings": 128,
            "rope_parameters": {"rope_theta": 123456.0},
            "tie_word_embeddings": False,
        },
        "rvq_depth_decoder": {
            "hidden_size": 64,
            "num_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "audio_vocab_size": 1024,
            "num_codebooks": 8,
            "max_position_embeddings": 16,
        },
        "condition_encoder": {
            "condition_hidden_dim": 64,
            "num_condition_layers": 8,
            "out_dim": 32,
            "input_sampling_rate": 24000,
            "input_hop_length": 960,
            "output_sampling_rate": 44100,
            "output_hop_length": 512,
        },
        "transformer": {
            "in_channels": 16,
            "condition_dim": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "ff_inner_dim": 64,
            "rotary_dim": 4,
            "fourier_embedding_dim": 16,
        },
        "vocoder": {
            "latent_channels": 16,
            "decoder_input_dim": 32,
            "decoder_hidden_dim": 64,
            "upsampling_ratios": [2, 2],
            "sampling_rate": 44100,
        },
    }
    for component, config in configs.items():
        path = root / component
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(config))


def test_source_configs_build_consistent_checkpoint(tmp_path) -> None:
    write_source_configs(tmp_path)
    config = checkpoint_config_from_source(tmp_path)

    assert config["components"]["language_model"]["rope_theta"] == 123456.0
    assert config["components"]["vocoder"]["upsampling_ratios"] == (2, 2)
    assert config["model"]["latent_hop_length"] == 4
    assert config["model"]["latent_channels"] == 16


def test_prepare_layout_copies_runtime_assets(tmp_path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_source_configs(source)
    tokenizer = source / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    (source / "LICENSE").write_text("license")

    prepare_checkpoint_layout(source, target)

    assert (target / "config.json").is_file()
    assert (target / "tokenizer" / "tokenizer.json").read_text() == "{}"
    assert (target / "LICENSE").read_text() == "license"
