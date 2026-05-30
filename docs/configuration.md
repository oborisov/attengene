# Configuration

All AttenGene configuration is via environment variables. Copy
[`.env.example`](../.env.example) to `.env` and override the values
that apply to you - every variable has a sensible default for the
bundled Compose stack.

Variables grouped by subsystem:

## LLM backend

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_LOCAL_URL` | `http://host.docker.internal:8080/v1/chat/completions` | Where your LLM lives. Any OpenAI-compatible chat-completions endpoint. |
| `BACKEND_LOCAL_MODEL` | `auto` | Upstream model name. `auto` calls `GET <BACKEND_LOCAL_URL>/models` once at startup and pins the result. |
| `LLM_TIMEOUT` | `120.0` | HTTP timeout (seconds) for LLM calls. |
| `LLM_MAX_TOKENS` | `2048` | Max tokens generated per response. |
| `BACKEND_MISTRAL_URL` | `https://api.mistral.ai/v1/chat/completions` | Mistral cloud endpoint. |
| `BACKEND_MISTRAL_KEY` | unset | When unset, `attengene-mistral` is **not** advertised on `/v1/models`. |
| `BACKEND_MISTRAL_MODEL` | `mistral-large-latest` | Upstream Mistral model id. |

Cloud backends send your chats off the host - the upstream provider
sees your queries and any retrieved snippets included in the prompt.
If that matters for your data, use the local backend.

## Embeddings server

AttenGene calls an HTTP embeddings server hosting a 1024-dim model
(default `BAAI/bge-large-en-v1.5`). The expected URL shape matches
Hugging Face's [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
(TEI) `/embed` endpoint - so any TEI server, or any service that
mimics that shape (including the one bundled in this repo), works
without code changes. The Compose stack ships one and wires it up
automatically; override `EMBEDDINGS_URL` to point at your own.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDINGS_URL` | `http://embeddings:8081/embed` in Compose, unset otherwise | The `/embed` endpoint to call. Required. |
| `EMBEDDINGS_TIMEOUT` | `10.0` | HTTP timeout (seconds). |
| `EMBEDDINGS_BATCH_SIZE` | `32` (app side), `256` (server side) | Documents per HTTP call. Bump the app-side value to `256-512` when pointing at a GPU server. |

### Embeddings server (when self-hosting the GPU container)

These are read by `app/embeddings_server.py` inside the embeddings
container, not by the API:

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDINGS_MODEL` | `BAAI/bge-large-en-v1.5` | Sentence-transformers model id. |
| `EMBEDDINGS_DEVICE` | `cuda` (GPU image), `cpu` (CPU image) | Compute device. |
| `EMBEDDINGS_DTYPE` | `float16` (GPU), `float32` (CPU) | Numerical precision. fp16 needs CUDA. |
| `EMBEDDINGS_BATCH_SIZE` | `256` | Server-side batch on the GPU. |
| `EMBEDDINGS_HOST` | `0.0.0.0` | Bind address. |
| `EMBEDDINGS_PORT` | `8081` | Bind port. |

## Database

| Variable | Default | Purpose |
|---|---|---|
| `DB_HOST` | `db` in Compose, `127.0.0.1` otherwise | PostgreSQL host. |
| `DB_PORT` | `5432` | PostgreSQL port. |
| `DB_NAME` | `attengene` | Database name. |
| `DB_USER` | `attengene` | Role. |
| `DB_PASSWORD` | `changeme` | **Override before exposing the API.** |

## Authentication

| Variable | Default | Purpose |
|---|---|---|
| `ATTENGENE_API_KEY` | (empty) | When set, `/v1/*` requires `Authorization: Bearer <key>`. Empty disables auth (fine for local dev). |

## Optional / advanced

| Variable | Default | Purpose |
|---|---|---|
| `NCBI_API_KEY` | unset | PubMed E-utilities API key. Only used by `app/pubmed.py`; raises NCBI's rate cap from 3 req/s to 10 req/s. |
