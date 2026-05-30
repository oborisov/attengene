# Contributing

This is an early-stage project. The codebase is small enough that
"read the code" is still the right move - this document is a pointer
to the bits that change most often.

## Dev setup

```bash
git clone https://codeberg.org/oborisov/attengene.git
cd attengene

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env to point at your local DB / LLM / embeddings server.

docker compose up -d db   # PostgreSQL + pgvector only

uvicorn app.main:app --reload
```

For an end-to-end stack including the API and embeddings containers,
run `docker compose up -d` instead - see the
[main README](../README.md).

## Tests

The test suite uses the standard library `unittest` runner:

```bash
python -m unittest discover tests
```

Individual test files:

| File | What it covers |
|---|---|
| `tests/test_embeddings.py` | The shared embedding client (HTTP transport). |
| `tests/test_embeddings_server.py` | The bundled embeddings FastAPI server. |
| `tests/test_llm_dispatch.py` | Multi-backend LLM dispatch. |
| `tests/test_models_endpoint.py` | `/v1/models` advertisement logic. |

CI is not yet configured for the public repo - run tests locally
before opening a PR.

## Code style

- Hyphens (`-`), not em-dashes or en-dashes, in all prose and code
  comments.
- Avoid one-line abbreviations of identifiers (`embed` for
  `embedding`, etc.) unless they already match the surrounding
  codebase.
- Default to writing no comments. When a comment is necessary,
  explain **why**, not **what** - the code already says what it
  does.
- Don't add backwards-compatibility shims for code paths that haven't
  shipped. Just change the code.

## License of contributions

By submitting a contribution (PR, patch, issue with attached code) you
agree that it is licensed under the same terms as the rest of the
project (Apache 2.0). No separate CLA.

## Filing issues and PRs

- Issues: describe the symptom, the environment (host OS, Compose vs.
  bare-metal, GPU vs. CPU), and the smallest reproduction you can
  produce.
- PRs: keep them small. One logical change per PR. Tests welcome but
  not required for first-time contributions.

## What needs help

The roadmap, roughly in priority order:

1. OMIM ingestion (schema + indexer + retriever).
2. gnomAD allele-frequency lookups (probably as a runtime annotation
   on retrieved variants rather than a separate indexed corpus).
3. PubMed semantic search.
4. Refining the query router as the number of corpora grows.
5. Insert-path performance work in the indexers (COPY FROM BINARY +
   temp-table merge - see the in-tree handover notes).

Pick one and open an issue to discuss the approach before writing
code.
