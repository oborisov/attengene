# Indexing

AttenGene queries an indexed corpus stored in pgvector. This document
covers loading the three currently-supported knowledge bases:

- **ClinVar** - ~326k pathogenic / likely-pathogenic variants
- **GeneReviews** - ~800 condition articles, chunked by section
- **NephroGenetics** - curated nephrology gene-phenotype table

OMIM, gnomAD, and PubMed are on the roadmap and not yet wired up.

## Starter dataset (5-minute quickstart)

For first-time users, the ClinVar indexer ships with a curated
25-gene starter set covering hereditary cancer, Lynch syndrome,
inherited cardiac disease, inherited kidney disease, and
hemoglobinopathies. ~30k variants total.

```bash
docker compose exec api \
    python scripts/index_clinvar.py --starter-dataset
```

The flag auto-downloads `variant_summary.txt.gz` from NCBI on first
run (~110 MB compressed, ~420 MB raw). With the bundled GPU
embeddings profile the index builds in a few minutes; with CPU
embeddings expect 30-45 minutes.

The starter gene list lives in `STARTER_GENES` at the top of
`scripts/index_clinvar.py`. To change it, edit the constant - it is
deliberately short and auditable.

## Full ClinVar

```bash
docker compose exec api \
    python scripts/index_clinvar.py
```

Without `--starter-dataset` the indexer ingests the full ClinVar
feed, filtered to pathogenic / likely-pathogenic / mixed entries on
GRCh38. Roughly 326k variants. Indexing time depends on embedding
throughput - the bundled GPU profile finishes in 5-15 minutes; CPU
embeddings will take several hours.

### Useful flags

| Flag | Purpose |
|---|---|
| `--starter-dataset` | Index the 25-gene starter set. Auto-downloads ClinVar if missing. |
| `--genes BRCA1,MLH1,...` | Limit indexing to specific genes. |
| `--all-significance` | Include benign, likely-benign, and VUS (default is pathogenic-only). |
| `--limit N` | Stop after N variants - useful for smoke tests. |
| `--no-clear` | Append instead of truncating the table first. |
| `--dry-run` | Parse and embed but don't write to the database. |
| `--batch-size N` | Documents per embed call. Default 100; bump to 500 with a GPU embeddings server. |
| `--clinvar-file PATH` | Override the default ClinVar file location. |

## GeneReviews

The GeneReviews indexer takes NXML files (NCBI Bookshelf
distribution) as positional arguments:

```bash
docker compose exec api \
    python scripts/index_genereviews.py data/genereviews/*.nxml
```

### Useful flags

| Flag | Purpose |
|---|---|
| `--genes BRCA1,MLH1,...` | Index only articles covering these genes. |
| `--section-types clinical,molecular,...` | Index only specified section types. |
| `--clear` | Truncate the GeneReviews table before indexing. |
| `--clear-all` | Truncate everything in the database. **Destructive.** |
| `--dry-run` | Parse only. |
| `--batch-size N` | Embedding batch size. |

Each article is chunked by section. ~800 articles, ~5k-15k chunks
depending on chunking strategy.

## NephroGenetics

```bash
docker compose exec api \
    python scripts/index_nephrogenetics.py
```

Small (a few hundred rows), finishes in seconds even on CPU
embeddings.

### Useful flags

| Flag | Purpose |
|---|---|
| `--batch-size N` | Embedding batch size. Default 50. |
| `--no-clear` | Append instead of truncating. |

## Where the data lives

```
data/
  clinvar/variant_summary.txt.gz   # NCBI tab-delimited dump
  genereviews/*.nxml               # NCBI Bookshelf XML
  nephrogenetics/*.tsv             # curated table
```

`data/` is mounted into the API container by the Compose stack, so
files dropped there are visible inside the container. The auto-download
in `--starter-dataset` writes to `data/clinvar/`.

## Embeddings throughput

Indexing speed is dominated by embedding throughput, not database
inserts. The GPU embeddings server (`docker compose --profile gpu
up -d`) gives a 20-30x speedup over CPU embeddings on a
mid-range NVIDIA card. The bundled CPU container works fine for the
starter dataset; for the full corpus you'll want a GPU or a hosted
embeddings endpoint.

## Database schema

The pgvector schema is in `sql/init.sql` (variants),
`sql/genereviews_schema.sql`, and `sql/audit_tables.sql`. The Compose
stack loads all three on first boot via the postgres
`/docker-entrypoint-initdb.d/` mechanism.
