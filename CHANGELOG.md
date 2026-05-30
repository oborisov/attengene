# Changelog

## v0.1.0 - 2026-05-30

Initial public release.

- FastAPI backend exposing an OpenAI-compatible `/v1/chat/completions`
  and `/v1/models` surface
- Hybrid retrieval (pg_trgm lexical + pgvector semantic) over ClinVar
- Semantic retrieval over GeneReviews and NephroGenetics
- Multi-backend LLM dispatch: a local OpenAI-compatible endpoint
  (`attengene-local`) and an optional Mistral cloud backend
  (`attengene-mistral`, opt-in via `BACKEND_MISTRAL_KEY`)
- Bundled embeddings server (BAAI/bge-large-en-v1.5, 1024-dim) with
  CPU and CUDA images
- Audit logging, citation post-processing, pre/post guardrails
- Indexing scripts for ClinVar (with a 25-gene starter dataset),
  GeneReviews (NXML), and NephroGenetics
- Docker Compose stack covering database, embeddings, and API; LLM is
  intentionally external
