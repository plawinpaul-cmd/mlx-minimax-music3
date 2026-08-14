"""Native Apple MLX inference for MiniMax Music 3."""

from .config import GenerationConfig, ModelConfig, OptimizationConfig

__all__ = [
    "GenerationConfig",
    "ModelConfig",
    "OptimizationConfig",
    "generate",
    "load_model",
]
__version__ = "0.2.0"


def load_model(model, *, optimization_config: OptimizationConfig | None = None):
    from .checkpoint import load_pipeline

    return load_pipeline(model, optimization_config=optimization_config)


def generate(model, prompt: str, lyrics: str, **kwargs):
    config = GenerationConfig(**kwargs)
    result = model.generate(prompt, lyrics, config)
    return result.audio, result.sampling_rate
