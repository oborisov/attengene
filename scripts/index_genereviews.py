#!/usr/bin/env python3
"""
Index GeneReviews into pgvector for RAG retrieval.

Parses NXML files, generates embeddings, and stores in PostgreSQL.

Usage:
    # Index all articles from tarball
    python scripts/index_genereviews.py data/gene_NBK1116.tar.gz

    # Index specific NXML files
    python scripts/index_genereviews.py data/gene_NBK1116/hnpcc.nxml data/gene_NBK1116/brca1.nxml

    # Index with filters
    python scripts/index_genereviews.py data/gene_NBK1116.tar.gz --genes BRCA1,MLH1,CFTR

    # Dry run (parse only, no DB)
    python scripts/index_genereviews.py data/gene_NBK1116.tar.gz --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genereviews_parser import parse_nxml, parse_tarball, GeneReviewsSection

# Database configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attengene")
DB_USER = os.getenv("DB_USER", "attengene")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")

# Embedding configuration. The model and HTTP transport live in
# app/embeddings.py; this script only owns the documents-per-encode_batch
# boundary.
BATCH_SIZE = 32


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def create_schema(conn):
    """Create the database schema if it doesn't exist."""
    schema_path = Path(__file__).parent.parent / "sql" / "genereviews_schema.sql"
    if schema_path.exists():
        with conn.cursor() as cur:
            cur.execute(schema_path.read_text())
        conn.commit()
        print("Schema created/verified", file=sys.stderr)
    else:
        print(f"WARNING: Schema file not found: {schema_path}", file=sys.stderr)


def clear_existing_data(conn, shortnames: list[str] = None):
    """Clear existing data for specific articles or all."""
    with conn.cursor() as cur:
        if shortnames:
            cur.execute(
                "DELETE FROM genereviews_chunks WHERE shortname = ANY(%s)",
                (shortnames,)
            )
            print(f"Cleared {cur.rowcount} existing chunks for: {shortnames}", file=sys.stderr)
        else:
            cur.execute("DELETE FROM genereviews_chunks")
            print(f"Cleared all {cur.rowcount} existing chunks", file=sys.stderr)
    conn.commit()


def insert_chunks(conn, sections: list[tuple]):
    """Insert chunks with embeddings into database."""
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO genereviews_chunks (
                nbk_id, shortname, article_title, condition_name, gene_symbols,
                section_id, section_title, section_path, section_type,
                chunk_text, chunk_index, char_count, token_estimate,
                retired, embedding
            ) VALUES %s
            """,
            sections,
            template="""(
                %(nbk_id)s, %(shortname)s, %(article_title)s, %(condition_name)s, %(gene_symbols)s,
                %(section_id)s, %(section_title)s, %(section_path)s, %(section_type)s,
                %(chunk_text)s, %(chunk_index)s, %(char_count)s, %(token_estimate)s,
                %(retired)s, %(embedding)s
            )"""
        )
    conn.commit()


def _section_to_row(sec: GeneReviewsSection, emb: list[float]) -> dict:
    return {
        "nbk_id": sec.nbk_id,
        "shortname": sec.shortname,
        "article_title": sec.article_title,
        "condition_name": sec.condition_name,
        "gene_symbols": sec.gene_symbols,
        "section_id": sec.section_id,
        "section_title": sec.section_title,
        "section_path": sec.section_path,
        "section_type": sec.section_type,
        "chunk_text": sec.text,
        "chunk_index": sec.chunk_index,
        "char_count": sec.char_count,
        "token_estimate": sec.token_estimate,
        "retired": sec.retired,
        "embedding": emb,
    }


def index_sections(conn, sections: list[GeneReviewsSection], batch_size: int = BATCH_SIZE):
    """Index sections with embeddings."""
    from tqdm import tqdm
    from app.embeddings import EmbedInsertPipeline

    total_indexed = 0

    with tqdm(total=len(sections), desc="Indexing") as pbar:
        def sink(batch: list[GeneReviewsSection], embeddings: list[list[float]]) -> None:
            nonlocal total_indexed
            rows = [_section_to_row(sec, emb) for sec, emb in zip(batch, embeddings)]
            insert_chunks(conn, rows)
            total_indexed += len(batch)
            pbar.update(len(batch))

        with EmbedInsertPipeline(sink=sink) as pipe:
            batch: list[GeneReviewsSection] = []
            for section in sections:
                batch.append(section)
                if len(batch) >= batch_size:
                    pipe.submit(batch, [s.text for s in batch])
                    batch = []
            if batch:
                pipe.submit(batch, [s.text for s in batch])

    return total_indexed


def filter_sections(sections, gene_filter: set[str] = None, section_types: set[str] = None):
    """Filter sections by gene symbols or section types."""
    for section in sections:
        # Gene filter
        if gene_filter:
            section_genes = set(section.gene_symbols)
            if not section_genes.intersection(gene_filter):
                continue

        # Section type filter
        if section_types and section.section_type not in section_types:
            continue

        yield section


def main():
    parser = argparse.ArgumentParser(description="Index GeneReviews into pgvector")
    parser.add_argument("inputs", nargs="+", type=Path, help="NXML files or tarball")
    parser.add_argument("--genes", type=str, help="Comma-separated gene symbols to filter")
    parser.add_argument("--section-types", type=str, help="Comma-separated section types")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before indexing")
    parser.add_argument("--clear-all", action="store_true", help="Clear ALL existing data")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't store")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Embedding batch size")

    args = parser.parse_args()

    # Parse filters
    gene_filter = set(args.genes.split(",")) if args.genes else None
    section_types = set(args.section_types.split(",")) if args.section_types else None

    # Collect all sections
    print("Parsing GeneReviews...", file=sys.stderr)
    all_sections = []

    # Expand any directory arguments into the .nxml files they contain, so
    # `index_genereviews.py data/gene_NBK1116/` works the same as passing the
    # glob. (A bare directory used to silently parse 0 sections.)
    expanded_inputs: list[Path] = []
    for input_path in args.inputs:
        if input_path.is_dir():
            nxml_files = sorted(input_path.glob("*.nxml"))
            if not nxml_files:
                print(f"WARNING: no .nxml files in directory {input_path}", file=sys.stderr)
            expanded_inputs.extend(nxml_files)
        else:
            expanded_inputs.append(input_path)

    for input_path in expanded_inputs:
        if input_path.suffix == ".gz":
            # Tarball - use quiet mode since we show final count
            for section in parse_tarball(input_path, quiet=True):
                all_sections.append(section)
        elif input_path.suffix == ".nxml":
            # Single file
            for section in parse_nxml(input_path):
                all_sections.append(section)
        else:
            print(f"Skipping unknown file type: {input_path}", file=sys.stderr)

    print(f"Parsed {len(all_sections)} sections", file=sys.stderr)

    # Apply filters
    if gene_filter or section_types:
        all_sections = list(filter_sections(all_sections, gene_filter, section_types))
        print(f"After filtering: {len(all_sections)} sections", file=sys.stderr)

    if args.dry_run:
        print("\n=== DRY RUN - No data stored ===", file=sys.stderr)
        # Show sample
        for i, sec in enumerate(all_sections[:5]):
            print(f"\n[{i+1}] {sec.article_title} > {sec.section_path}")
            print(f"    Type: {sec.section_type}, Tokens: ~{sec.token_estimate}")
            print(f"    Genes: {sec.gene_symbols}")
        if len(all_sections) > 5:
            print(f"\n... and {len(all_sections) - 5} more sections", file=sys.stderr)
        return

    # Guard against the destructive-clear footgun: if parsing produced nothing
    # (bad path, empty dir, wrong file type), do NOT clear - otherwise a
    # --clear/--clear-all run wipes the live index and refills it with nothing.
    if not all_sections:
        print("ERROR: parsed 0 sections - refusing to clear/index. "
              "Check the input path(s).", file=sys.stderr)
        return 1

    # Initialize
    print("Connecting to database...", file=sys.stderr)
    conn = get_db_connection()
    create_schema(conn)

    if args.clear_all:
        clear_existing_data(conn)
    elif args.clear:
        shortnames = list(set(s.shortname for s in all_sections))
        clear_existing_data(conn, shortnames)

    from app.embeddings import EMBEDDINGS_URL
    if not EMBEDDINGS_URL:
        print("Error: EMBEDDINGS_URL is not set. Point it at your "
              "embeddings server's /embed endpoint.", file=sys.stderr)
        return 1
    print(f"Embeddings server: {EMBEDDINGS_URL}", file=sys.stderr)

    print("Indexing sections...", file=sys.stderr)
    total = index_sections(conn, all_sections, args.batch_size)

    print(f"\n=== INDEXING COMPLETE ===", file=sys.stderr)
    print(f"Total sections indexed: {total}", file=sys.stderr)

    # Show stats
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), SUM(token_estimate) FROM genereviews_chunks")
        count, tokens = cur.fetchone()
        print(f"Database now contains: {count} chunks, ~{tokens:,} tokens", file=sys.stderr)

    conn.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
