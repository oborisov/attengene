"""
FastAPI server that hosts BAAI/bge-large-en-v1.5 via sentence-transformers + torch.

Drop-in replacement for the HuggingFace text-embeddings-inference (TEI) image,
written because TEI/Candle's CUDA path on Blackwell sm_120 runs at ~70 emb/s
while direct sentence-transformers+torch fp16 does ~1200 emb/s on the same
card. See ../handover-tei-too-slow-20260521.md for the full picture.

Two POST endpoints, same shapes the existing client code (`app/embeddings.py`)
and any future OpenAI-compatible consumer expects:

  POST /embed
    request:  {"inputs": ["text", "text", ...]}
    response: [[1024 floats], [1024 floats], ...]
    (bare list - matches TEI's `/embed` shape)

  POST /v1/embeddings
    request:  {"input": ["text", ...], "model": "..."}
              (also accepts {"input": "single text"} per OpenAI's spec)
    response: {"data": [{"object": "embedding", "embedding": [...], "index": int}],
               "model": "...", "object": "list",
               "usage": {"prompt_tokens": 0, "total_tokens": 0}}

  GET /health
    503 until the model has finished loading onto CUDA; 200 with
    {"status": "ok"} after.

Configuration is env-driven so the systemd unit / Quadlet on the infra side
can tune without code changes:

  EMBEDDINGS_MODEL          default BAAI/bge-large-en-v1.5
  EMBEDDINGS_DEVICE         default cuda (must be CUDA; CPU is rejected to
                            avoid silently regressing to 1-2 emb/s)
  EMBEDDINGS_DTYPE          default float16
  EMBEDDINGS_BATCH_SIZE     default 256 - inner GPU chunk size; large client
                            POSTs are split into chunks of this size before
                            hitting model.encode()
  EMBEDDINGS_HOST           default 0.0.0.0
  EMBEDDINGS_PORT           default 8081

Run with `python scripts/run_embeddings_server.py` or
`uvicorn app.embeddings_server:app --host ... --port ...`.

Threading / concurrency: this module deliberately serializes GPU access with
a single asyncio lock. Concurrent requests do not run encode() in parallel
inside one process - the GPU is the bottleneck and parallelism inside a
single CUDA context only adds overhead. For request-level parallelism on the
serving box, run multiple uvicorn workers (each holds its own model copy on
the GPU); the infra side decides whether the VRAM headroom is worth it.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("embeddings_server")


EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDINGS_DEVICE = os.getenv("EMBEDDINGS_DEVICE", "cuda")
EMBEDDINGS_DTYPE = os.getenv("EMBEDDINGS_DTYPE", "float16")
EMBEDDINGS_BATCH_SIZE = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "256"))


# Module-level state populated by the lifespan hook so request handlers can
# access them without indirection. _model is None until /health flips to 200.
_model = None
_model_lock: Optional[asyncio.Lock] = None
_model_ready = False


def _torch_dtype(name: str):
    """Map a string env var to a torch dtype. Imports torch lazily."""
    import torch
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(
            f"Unsupported EMBEDDINGS_DTYPE={name!r}; "
            f"choose one of {sorted(table)}"
        )
    return table[name]


def _load_model():
    """
    Synchronously load the model onto the configured device. Called from the
    lifespan hook before the first request is accepted. Raises on any
    failure - no silent CPU fallback.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if EMBEDDINGS_DEVICE != "cuda":
        raise RuntimeError(
            f"EMBEDDINGS_DEVICE={EMBEDDINGS_DEVICE!r} is not 'cuda'. "
            f"This server refuses CPU on purpose - CPU inference for "
            f"bge-large-en-v1.5 runs at ~1-2 emb/s and would silently "
            f"regress production. Set EMBEDDINGS_DEVICE=cuda or use the "
            f"in-process path in app/embeddings.py."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() == False. Either CUDA isn't built "
            "into this torch install, or the driver isn't visible. "
            "Check `nvidia-smi` and the torch+cu* wheel."
        )

    dtype = _torch_dtype(EMBEDDINGS_DTYPE)
    t0 = time.perf_counter()
    model = SentenceTransformer(
        EMBEDDINGS_MODEL,
        device=EMBEDDINGS_DEVICE,
        model_kwargs={"torch_dtype": dtype},
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded %s on %s dtype=%s in %.1fs (CUDA device %d: %s)",
        EMBEDDINGS_MODEL,
        EMBEDDINGS_DEVICE,
        EMBEDDINGS_DTYPE,
        elapsed,
        torch.cuda.current_device(),
        torch.cuda.get_device_name(),
    )
    return model


def _encode(texts: list[str]) -> list[list[float]]:
    """
    Run the model on a list of texts. Caller has already taken the GPU lock,
    so this is the only place that touches the model.

    Splits into chunks of EMBEDDINGS_BATCH_SIZE before each model.encode()
    call so a 5000-doc POST doesn't try to load the whole batch onto the
    GPU at once.
    """
    if not texts:
        return []
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDINGS_BATCH_SIZE):
        chunk = texts[start:start + EMBEDDINGS_BATCH_SIZE]
        vecs = _model.encode(
            chunk,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        out.extend(v.tolist() for v in vecs)
    return out


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _model, _model_lock, _model_ready
    _model_lock = asyncio.Lock()
    _model = _load_model()
    _model_ready = True
    try:
        yield
    finally:
        _model_ready = False


app = FastAPI(
    title="AttenGene embeddings",
    description=(
        "Sentence-transformers-backed embeddings server hosting "
        f"{EMBEDDINGS_MODEL}. Drop-in replacement for TEI's /embed and "
        "OpenAI's /v1/embeddings."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------- Request / response models ----------


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1)


class OpenAIEmbeddingsRequest(BaseModel):
    input: Union[str, list[str]]
    model: Optional[str] = None


class OpenAIEmbeddingItem(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAIEmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIEmbeddingItem]
    model: str
    usage: OpenAIUsage = OpenAIUsage()


# ---------- Routes ----------


@app.get("/health")
async def health():
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading"},
        )
    return {"status": "ok"}


@app.post("/embed")
async def embed(req: EmbedRequest):
    if not _model_ready:
        raise HTTPException(503, "model not ready")
    t0 = time.perf_counter()
    async with _model_lock:
        # GPU work is sync; run in a thread so we don't block the event loop.
        vecs = await asyncio.to_thread(_encode, req.inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "/embed n=%d elapsed=%.0fms (%.0f emb/s)",
        len(req.inputs),
        elapsed_ms,
        (len(req.inputs) / (elapsed_ms / 1000)) if elapsed_ms else 0.0,
    )
    return vecs


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingsResponse)
async def openai_embeddings(req: OpenAIEmbeddingsRequest):
    if not _model_ready:
        raise HTTPException(503, "model not ready")
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(400, "input must be non-empty")
    t0 = time.perf_counter()
    async with _model_lock:
        vecs = await asyncio.to_thread(_encode, inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "/v1/embeddings n=%d elapsed=%.0fms (%.0f emb/s)",
        len(inputs),
        elapsed_ms,
        (len(inputs) / (elapsed_ms / 1000)) if elapsed_ms else 0.0,
    )
    return OpenAIEmbeddingsResponse(
        data=[
            OpenAIEmbeddingItem(embedding=v, index=i)
            for i, v in enumerate(vecs)
        ],
        model=req.model or EMBEDDINGS_MODEL,
    )
