from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .audio import wav_bytes
from .checkpoint import load_pipeline
from .config import GenerationConfig


class SpeechRequest(BaseModel):
    model: str
    input: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    response_format: Literal["wav"] = "wav"
    seed: int = 0
    max_new_tokens: int = Field(default=1500, ge=1, le=9000)
    stream: bool = False


def create_app(
    model_id: str,
    *,
    pipeline_loader: Callable = load_pipeline,
) -> FastAPI:
    app = FastAPI(title="MiniMax Music 3 MLX", version="0.1.0")
    state: dict[str, object] = {}
    load_lock = threading.Lock()
    generation_lock = threading.Lock()

    def pipeline():
        if "pipeline" not in state:
            with load_lock:
                if "pipeline" not in state:
                    state["pipeline"] = pipeline_loader(model_id)
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model": model_id, "loaded": "pipeline" in state}

    @app.post("/v1/audio/speech")
    def speech(request: SpeechRequest) -> Response:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming generation is not supported")
        active = pipeline()
        frame_rate = active.model_config.frame_rate
        generation = GenerationConfig(
            audio_duration=request.max_new_tokens / frame_rate,
            seed=request.seed,
        )
        try:
            with generation_lock:
                result = active.generate(request.instructions, request.input, generation)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return Response(
            content=wav_bytes(result.audio, result.sampling_rate),
            media_type="audio/wav",
            headers={
                "X-MiniMax-Music3-Frames": str(result.num_frames),
                "X-MiniMax-Music3-Chunks": str(result.num_chunks),
            },
        )

    return app

