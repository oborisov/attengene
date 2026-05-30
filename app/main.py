"""
FastAPI application for AttenGene.

Orchestration only - no business logic in routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.llm import resolve_local_upstream_model
from app.routes_openai import router as openai_router

app = FastAPI(
    title="AttenGene",
    description="Clinical Genetics AI Assistant - ClinVar variant evidence exploration",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)


@app.on_event("startup")
async def _resolve_backend_models() -> None:
    """One-shot: resolve `BACKEND_LOCAL_MODEL=auto` from llama-server /v1/models."""
    await resolve_local_upstream_model()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
