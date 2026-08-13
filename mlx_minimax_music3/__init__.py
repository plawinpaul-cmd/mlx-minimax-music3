"""Native Apple MLX inference for MiniMax Music 3."""

from .config import GenerationConfig, ModelConfig

__all__ = ["GenerationConfig", "ModelConfig", "generate", "load_model"]
__version__ = "0.1.0.dev0"


def load_model(model):
    from .checkpoint import load_pipeline

    return load_pipeline(model)


def generate(model, prompt: str, lyrics: str, **kwargs):
    config = GenerationConfig(**kwargs)
    result = model.generate(prompt, lyrics, config)
    return result.audio, result.sampling_rate
