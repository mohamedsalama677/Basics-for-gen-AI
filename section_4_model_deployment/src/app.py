"""FastAPI app: /health, /generate (blocking), /generate/stream (SSE)."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, MODEL_FILE
from model import LlamaService

_service: LlamaService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = LlamaService()
    yield
    _service = None


app = FastAPI(
    title="Section 4 — LLM Serving",
    description=(
        "Small CPU-portable LLM behind a FastAPI service. "
        f"Backing model: {MODEL_FILE} (Qwen2.5-0.5B GGUF Q4_K_M via llama.cpp)."
    ),
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=1024)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    latency_ms: float


def _get_service() -> LlamaService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _service


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok" if _service is not None else "loading", "model": MODEL_FILE}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    svc = _get_service()
    t0 = time.perf_counter()
    text, tokens = svc.generate(req.prompt, req.max_tokens, req.temperature)
    latency_ms = (time.perf_counter() - t0) * 1000
    return GenerateResponse(text=text, tokens=tokens, latency_ms=round(latency_ms, 1))


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    svc = _get_service()

    async def event_source():
        try:
            async for token in svc.stream(req.prompt, req.max_tokens, req.temperature):
                # SSE frame: data: <token>\n\n
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # surfaces as an SSE error frame, not a 500
            yield f"event: error\ndata: {exc!s}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
