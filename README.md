# AttenGene

A retrieval-augmented chat interface over public clinical-genetics
knowledge bases (ClinVar, GeneReviews, NephroGenetics; OMIM, gnomAD, and
PubMed are on the roadmap). Bring your own LLM and embeddings server;
AttenGene wires them up with a vector database, hybrid retrieval, source
citations, and an OpenAI-compatible chat endpoint.

> **AttenGene is a scientific exploration tool, not a medical device.**
>
> It surfaces information from public knowledge bases and presents it
> through an LLM. It is intended for research, learning, and literature
> exploration. **It is not for diagnostic use, not for treatment
> decisions, and not a substitute for advice from a qualified
> professional.** The software is provided as-is under the Apache 2.0
> License with no warranty of any kind.

## What you get

- **OpenAI-compatible API.** Drop AttenGene into any OpenAI-API client
  (Open WebUI, LibreChat, your own app) as a model backend. It speaks
  `/v1/chat/completions` and `/v1/models`.
- **Hybrid retrieval over pgvector.** Lexical (pg_trgm) for exact
  variant notation (HGVS, RS IDs); semantic (BGE-large-en-v1.5
  embeddings) for phenotype and free-text queries. Per-database routing
  picks the right strategy.
- **Source citations on every answer.** Variant IDs, GeneReview
  sections, and source URLs are returned alongside the model output.
- **Stays on your hardware.** No external services required at runtime
  once databases are indexed. Bring your own LLM (llama.cpp, Ollama,
  vLLM, ...) and the whole stack runs offline.

## Quickstart (5 minutes)

```bash
# 1. Clone
git clone https://codeberg.org/oborisov/attengene.git
cd attengene

# 2. Configure
cp .env.example .env
# Edit .env if you want non-default DB credentials or a remote LLM.

# 3. Start a local LLM (one-time setup, your choice)
#    Pick one - llama.cpp, Ollama, or any other OpenAI-compatible server.
#    Example with Ollama:
#      ollama pull qwen2.5:7b
#      ollama serve

# 4. Bring up AttenGene
docker compose up -d

# 5. Index the starter dataset (BRCA1/2, Lynch syndrome genes, TP53, ...)
docker compose exec api \
    python scripts/index_clinvar.py --starter-dataset

# 6. Query
curl -s http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
      "model": "attengene-local",
      "messages": [{"role": "user", "content": "What is known about BRCA1 c.5266dupC?"}]
    }'
```

If you have an NVIDIA GPU, add `--profile gpu` to step 4 - embeddings
will run in fp16 on CUDA instead of fp32 on CPU (roughly 20-30x
throughput on a mid-range GPU).

## Architecture

```
┌────────────────┐
│  Your chat UI  │      (Open WebUI, your own app, curl, ...)
│  (OpenAI API)  │
└────────┬───────┘
         │  /v1/chat/completions
         ▼
┌──────────────────────────────────────────────────────────┐
│  AttenGene API (FastAPI)                                 │
│                                                          │
│  - query classification + routing                        │
│  - hybrid retrieval (lexical + semantic)                 │
│  - guardrails, citation assembly, audit logging          │
└────────┬───────────────┬───────────────────┬─────────────┘
         │               │                   │
         ▼               ▼                   ▼
┌──────────────┐  ┌──────────────┐    ┌──────────────────┐
│  pgvector    │  │  Embeddings  │    │  LLM backend     │
│  (PostgreSQL)│  │  (BGE-large) │    │  (your choice)   │
└──────────────┘  └──────────────┘    └──────────────────┘
       ^                  ^                    ^
       │                  │                    │
       └─ in Compose ─────┘                    └─ external
```

The three boxes inside the dashed line ship in the bundled
`docker-compose.yml`. The LLM is intentionally external: you pick
the model and the server that fit your hardware and licensing
constraints.

## Bring your own data

The starter dataset (25 well-curated genes across hereditary cancer,
Lynch syndrome, inherited cardiac, inherited kidney, hematology, and
CF) gives you ~30k pathogenic variants - small enough to index in a
few minutes on a GPU embeddings server, large enough to be worth
querying. For the full deal:

| Source | What | How |
|--------|------|-----|
| ClinVar | ~326k pathogenic / likely-pathogenic variants | `scripts/index_clinvar.py` |
| GeneReviews | ~800 condition articles | `scripts/index_genereviews.py` |
| NephroGenetics | curated nephrology gene-phenotype table | `scripts/index_nephrogenetics.py` |

Full indexing takes 15-60 minutes depending on hardware, dominated by
embedding throughput. The bundled GPU profile (RTX 30+ class) does the
whole job in well under an hour.

## Configuration

All configuration is via environment variables - see
[`.env.example`](.env.example) for the canonical list.
The variables you'll most likely touch:

| Variable | Default | Purpose |
|----------|---------|---------|
| `BACKEND_LOCAL_URL` | `http://host.docker.internal:8080/v1/chat/completions` | Where your LLM lives |
| `BACKEND_LOCAL_MODEL` | `auto` | Upstream model name (`auto` queries `/models`) |
| `EMBEDDINGS_URL` | `http://embeddings:8081/embed` | Embeddings server (in-Compose by default) |
| `DB_PASSWORD` | `changeme` | Override before going to production |
| `ATTENGENE_API_KEY` | (empty) | Set to enable bearer-token auth |

## Documentation

- [`docs/architecture.md`](docs/architecture.md) - longer overview,
  including retrieval strategy per database
- [`docs/configuration.md`](docs/configuration.md) - the full env-var
  reference
- [`docs/indexing.md`](docs/indexing.md) - how to load your own data
- [`docs/api.md`](docs/api.md) - the FastAPI surface, including the
  OpenAI-compatible endpoints
- [`docs/contributing.md`](docs/contributing.md) - development setup,
  test suite, code style

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

AttenGene is built on top of - and is only useful because of - several
public knowledge bases:

- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (NCBI)
- [GeneReviews](https://www.ncbi.nlm.nih.gov/books/NBK1116/) (University
  of Washington, Seattle)
- The NephroGenetics resource

If you publish work that uses AttenGene, please cite the underlying
sources, not the wrapper.
