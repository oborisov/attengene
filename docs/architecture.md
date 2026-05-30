# Architecture

AttenGene is a thin FastAPI orchestration layer around three pluggable
backends:

- a **vector database** (pgvector on PostgreSQL) holding indexed
  knowledge-base content
- an **embeddings server** that turns text into 1024-dim vectors
- an **LLM** that speaks the OpenAI chat-completions shape

The orchestration layer owns query classification, retrieval, citation
assembly, audit logging, and the OpenAI-compatible HTTP surface that
clients talk to. The three backends are all swappable: you can point
them at containers shipped in the bundled Compose stack, at services
running elsewhere on your network, or at hosted APIs.

## Request flow

```
client
  │
  │  POST /v1/chat/completions { model, messages }
  ▼
┌─────────────────────────────────────────────────────┐
│ FastAPI (app/main.py + app/routes_openai.py)        │
│                                                     │
│   1. validate_query (app/guardrails.py)             │
│   2. route_and_retrieve (app/router.py)             │
│        │                                            │
│        ├─ classify intent (variant / phenotype /    │
│        │  GeneReview-shape / kidney-specific / ...) │
│        │                                            │
│        ├─ embed query                ───────────────┼──▶ embeddings server
│        │                                            │
│        └─ hybrid / semantic search   ───────────────┼──▶ pgvector
│                                                     │
│   3. build_augmented_messages (app/prompts.py)      │
│   4. generate / generate_stream      ───────────────┼──▶ LLM backend
│      (app/llm.py)                                   │
│   5. validate_response (guardrails)                 │
│   6. postprocess_citations (app/citations.py)       │
│   7. log_query (app/audit.py)                       │
└─────────────────────────────────────────────────────┘
  │
  ▼
client (OpenAI-shaped response with citations)
```

The same pipeline runs whether the LLM is local (llama-server, Ollama,
vLLM) or a hosted API. Only the final HTTP call out of step 4 differs.

## Retrieval strategy per database

| Database | Data shape | Search strategy | Why |
|---|---|---|---|
| ClinVar | structured variants, exact identifiers | hybrid: pg_trgm trigram + pgvector cosine | exact HGVS / RS-ID lookups need lexical; phenotype queries need semantic |
| GeneReviews | unstructured prose, sectioned | semantic (pgvector cosine) | clinical narrative needs embedding similarity, not substring matching |
| NephroGenetics | semi-structured gene-phenotype table | semantic (pgvector cosine) | phenotype queries benefit from embeddings; gene symbols are exact-matched as a fast path |

**Hybrid search** (`app/retrieval.py:retrieve_variants_hybrid`):
trigram similarity on `name` and `document` columns is combined with
pgvector cosine similarity. Both scores are min-max normalized over
the candidate set, then combined as
`alpha * semantic + (1 - alpha) * lexical` with a default
`alpha = 0.3` (lexical-weighted, because variant-name lookups dominate
clinical queries). ClinVar queries use hybrid by default.

## Why pgvector

A single PostgreSQL instance covers the vector store, the audit log,
and any future relational metadata, on a stack any sysadmin already
knows. Migration to a dedicated vector DB (Milvus, Qdrant, etc.) is
possible later but never has been necessary at the scales AttenGene
targets (sub-million-row corpora).

## Why the LLM is external

LLM licensing, hardware, and performance trade-offs change month to
month. Pinning a model into the container would tie release cadence
to upstream model updates and lock users into one inference engine.
Instead, AttenGene speaks the OpenAI `/v1/chat/completions` shape and
points at whatever endpoint you configure - the same code path works
for llama.cpp's llama-server, Ollama, vLLM, LM Studio, OpenAI,
Mistral, and most other inference servers. See
[`docs/configuration.md`](configuration.md) for the relevant env vars.

## Multi-backend dispatch

The OpenAI-compatible surface (`app/routes_openai.py`) exposes each
configured backend as a separate model id:

- `attengene-local` - always available; routes to whatever endpoint
  `BACKEND_LOCAL_URL` points at
- `attengene-mistral` - only advertised when `BACKEND_MISTRAL_KEY` is
  set; routes to the Mistral cloud API
- `attengene-claude` - placeholder for an Anthropic backend, not yet
  wired in

The RAG pipeline is identical across backends; only the final HTTP
call differs. Clients pick which model id to use; the `/v1/models`
endpoint advertises whatever is configured at runtime.

## Code layout

```
app/
  main.py                # FastAPI app, /health, mounts /v1/* router
  routes_openai.py       # /v1/models, /v1/chat/completions
  router.py              # query classification + dispatch to retrievers
  db.py                  # shared psycopg2 connection helper
  embeddings.py          # HTTP embedding client (TEI-shape)
  embeddings_server.py   # FastAPI server hosting BGE-large (TEI shape)
  retrieval.py           # ClinVar hybrid search
  retrieval_genereviews.py
  retrieval_nephrogenetics.py
  llm.py                 # OpenAI-compatible LLM client + multi-backend dispatch
  prompts.py             # system prompts, RAG prompt assembly
  guardrails.py          # pre- and post-validation
  citations.py           # citation extraction and post-processing
  audit.py               # query logging
  auth.py                # API-key middleware
  models.py              # domain Pydantic models
  openai_compat.py       # OpenAI API Pydantic models

scripts/
  index_clinvar.py
  index_genereviews.py
  index_nephrogenetics.py
  run_embeddings_server.py

sql/
  init.sql               # variants table + HNSW index
  genereviews_schema.sql
  audit_tables.sql
```

For environment configuration see
[`docs/configuration.md`](configuration.md); for indexing see
[`docs/indexing.md`](indexing.md); for the full API surface see
[`docs/api.md`](api.md).
