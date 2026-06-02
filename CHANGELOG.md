# Changelog

## v0.2.0 - 2026-06-02

- RAG prompt: responses now open with a 2-3 sentence ANSWER synthesis
  (condition + gene(s), evidence-only, ending with a source list) before
  the FINDINGS / GAPS audit trail
- RAG prompt: tightened FINDINGS to at most 6 bullets, one fact each, with
  non-matching conditions collapsed into a single rule-out line instead of
  one bullet per retrieved entry
- Local backend: optional `BACKEND_LOCAL_KEY` sends `Authorization: Bearer`
  on outbound calls (chat and the startup `/models` probe) for a
  llama-server running with `--api-key`; empty/unset keeps the no-auth path

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
