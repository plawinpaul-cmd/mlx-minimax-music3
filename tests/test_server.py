import io
from dataclasses import dataclass

import mlx.core as mx
import soundfile as sf
from fastapi.testclient import TestClient

from mlx_minimax_music3.config import ModelConfig
from mlx_minimax_music3.pipeline import GenerationResult
from mlx_minimax_music3.server import create_app


@dataclass
class FakePipeline:
    model_config: ModelConfig = ModelConfig()

    def generate(self, prompt, lyrics, generation):
        assert prompt == "warm acoustic pop"
        assert lyrics == "hello world"
        assert generation.max_frames(self.model_config) == 50
        return GenerationResult(
            audio=mx.zeros((1, 2, 200)),
            sampling_rate=44_100,
            num_frames=50,
            num_chunks=1,
        )


def test_health_is_lazy_and_speech_returns_wav() -> None:
    loads = []

    def loader(model_id):
        loads.append(model_id)
        return FakePipeline()

    client = TestClient(create_app("local-model", pipeline_loader=loader))
    health = client.get("/health")
    assert health.json() == {"status": "ok", "model": "local-model", "loaded": False}

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "minimax_ttm",
            "input": "hello world",
            "instructions": "warm acoustic pop",
            "max_new_tokens": 50,
            "seed": 7,
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-minimax-music3-frames"] == "50"
    values, rate = sf.read(io.BytesIO(response.content), always_2d=True)
    assert rate == 44_100
    assert values.shape == (200, 2)
    assert loads == ["local-model"]


def test_streaming_request_is_rejected_without_loading() -> None:
    client = TestClient(create_app("local-model", pipeline_loader=lambda _: FakePipeline()))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "minimax_ttm",
            "input": "hello",
            "instructions": "pop",
            "stream": True,
        },
    )
    assert response.status_code == 400

