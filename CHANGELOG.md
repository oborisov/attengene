# Changelog

## v0.3.1 - 2026-06-02

- Fix: GeneReviews retrieval no longer cites an unrelated gene's chapter.
  When a query names a gene that isn't in the indexed corpus, pure semantic
  search returned the densest "genetics prose" chunk (an off-target gene
  scoring ~0.55), which surfaced as a citation even when the answer said
  "no evidence". Retrieval is now anchored on gene symbols extracted from
  the query (passed as a gene filter), so off-target chapters can't leak
  in; absent genes return an empty GeneReviews result cleanly.
- Fix: raised the GeneReviews similarity floor from 0.3 to 0.6. On-target
  chunks score ~0.7-0.8 while noise tops out around 0.55, so the old 0.3
  threshold filtered nothing.

## v0.3.0 - 2026-06-02

- Fix: the API image now builds on a fresh clone. The `Dockerfile` copies
  `data/genereviews/`, but those lookup tables were gitignored, so
  `docker/podman compose up` failed at the `COPY data/genereviews/` step.
  The two factual NBK-id/gene/disease lookup tables (the demo data the API
  loads at runtime) are now committed; copyrighted GeneReviews prose stays
  out of the repo.
- Podman: all image references are fully qualified (`docker.io/...`) in the
  compose file and Dockerfiles, so the stack runs under stock Podman without
  an `unqualified-search-registries` entry in `registries.conf`. Quickstart
  documents the `podman compose` path.

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
