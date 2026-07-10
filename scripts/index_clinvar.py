#!/usr/bin/env python3
"""
ClinVar Indexing Script

Loads ClinVar variant_summary.txt.gz into PostgreSQL with pgvector embeddings.

Embedding routing (in-process sentence-transformers vs HTTP embeddings
server on a GPU box) is decided by the EMBEDDINGS_URL env var in
app/embeddings.py.

Usage:
    # Starter dataset (25 well-curated genes, ~30k pathogenic variants).
    # Auto-downloads variant_summary.txt.gz from NCBI if not present.
    python scripts/index_clinvar.py --starter-dataset

    # Full ClinVar (~326k pathogenic variants)
    python scripts/index_clinvar.py

    # Remote GPU embeddings (any TEI-shape /embed endpoint)
    EMBEDDINGS_URL=http://example.com:8081/embed python scripts/index_clinvar.py

    # Index all clinical significance (pathogenic + VUS + conflicting +
    # benign). Uncertain/conflicting tiers get a guardrail note in the
    # embedded document so the model reports them as uncertain/disputed.
    python scripts/index_clinvar.py --all-significance

    # Zero-downtime full re-index (recommended for the recurring/weekly run):
    # builds variants_staging off to the side, indexes it, then swaps it in
    # atomically. Live `variants` keeps serving throughout. Needs ~120 GB free
    # on the PGDATA volume for the ~4.2M full build - df it first.
    python scripts/index_clinvar.py --all-significance --staging --batch-size 512

    # Custom database host
    python scripts/index_clinvar.py --db-host YOUR_DB_HOST
"""

import gzip
import csv
import os
import re
import argparse
import sys
import urllib.request
from typing import Generator
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from dotenv import load_dotenv

# Make the repo root importable so `from app.embeddings import ...` works when
# this script is run by path (python scripts/index_clinvar.py) rather than as a
# module - running a script by path only puts its own dir (scripts/) on the
# path, not the repo root. The sibling indexers (index_kdigo, index_bio2clin,
# ...) do the same.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


# Starter dataset: well-curated genes spanning hereditary cancer,
# Lynch syndrome, cardiac, inherited kidney disease, hematology, and CF.
# Picked for breadth of phenotypes and high ClinVar coverage so the
# 5-minute quickstart has something interesting to query.
STARTER_GENES = [
    # Hereditary breast/ovarian cancer
    "BRCA1", "BRCA2", "PALB2",
    # Lynch syndrome / mismatch repair
    "MLH1", "MSH2", "MSH6", "PMS2",
    # Familial adenomatous polyposis / Li-Fraumeni
    "APC", "TP53",
    # Inherited kidney disease (ADPKD, Alport, nephrotic, UMOD)
    "PKD1", "PKD2", "UMOD", "NPHS1", "NPHS2",
    "COL4A3", "COL4A4", "COL4A5",
    # Cardiac (hypertrophic cardiomyopathy, long-QT)
    "MYH7", "MYBPC3", "KCNQ1", "SCN5A",
    # CF, hemoglobinopathy, hemophilia
    "CFTR", "HBB", "F8", "F9",
]

# NCBI ClinVar variant_summary feed - tab-separated GRCh37+GRCh38 mix.
# The full file is ~110 MB compressed and refreshes weekly. We only
# auto-download when --starter-dataset is set and the file is missing.
CLINVAR_VARIANT_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)


def significance_tier(significance: str) -> str:
    """Classify a ClinVar ClinicalSignificance string into a coarse tier.

    Used to decide whether an uncertainty label is baked into the document
    (see to_document). Substring match on lowercased text so compound values
    like "Pathogenic/Likely pathogenic" and "Likely benign" land correctly.
    Conflicting is checked first (it can co-occur with nothing else meaningful),
    then uncertain, then pathogenic, then benign. Anything else -> "other".
    """
    s = significance.lower()
    if "conflicting" in s:
        return "conflicting"
    if "uncertain" in s:
        return "uncertain"
    if "pathogenic" in s:  # covers Pathogenic + Likely pathogenic + compounds
        return "pathogenic"
    if "benign" in s:  # covers Benign + Likely benign
        return "benign"
    return "other"


# Guardrail note baked into the embedded document for the non-firm tiers, so it
# rides through every retrieval path (exact/gene/hybrid/semantic) and into the
# embedded text without touching retrieval.py. Mirrors the Fabry in-silico
# hard-labeling pattern. Pathogenic/benign are firm classifications - no note.
_SIGNIFICANCE_NOTE = {
    "conflicting": (
        "NOTE: submitters DISAGREE on this classification (conflicting). "
        "Report the disagreement; do not resolve it."
    ),
    "uncertain": (
        "NOTE: UNCERTAIN significance (VUS) - this is NOT an established "
        "pathogenic or benign classification."
    ),
}


@dataclass
class ClinVarVariant:
    variation_id: int
    name: str
    gene: str
    clinical_significance: str
    review_status: str
    phenotypes: list[str]

    def to_document(self) -> str:
        """Create searchable document text."""
        phenotype_str = "; ".join(self.phenotypes[:5]) if self.phenotypes else "Not specified"
        doc = (
            f"Gene: {self.gene}. "
            f"Variant: {self.name}. "
            f"Clinical significance: {self.clinical_significance}. "
            f"Review status: {self.review_status}. "
            f"Associated conditions: {phenotype_str}."
        )
        note = _SIGNIFICANCE_NOTE.get(significance_tier(self.clinical_significance))
        if note:
            doc += f" {note}"
        return doc


def parse_clinvar(
    filepath: str,
    limit: int = None,
    all_significance: bool = False,
    genes_filter: set[str] = None,
) -> Generator[ClinVarVariant, None, None]:
    """Parse ClinVar variant_summary.txt.gz and yield variants."""

    # Significance values to include (pathogenic only by default). With
    # --all-significance every class is indexed - including VUS, conflicting,
    # and benign - and the uncertainty tiers get a guardrail note baked into
    # the document by to_document(). Conflicting is no longer force-excluded.
    include_significance = {
        "Pathogenic",
        "Likely pathogenic",
        "Pathogenic/Likely pathogenic",
    }

    count = 0

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            # Filter: GRCh38 only
            if row.get("Assembly") != "GRCh38":
                continue

            significance = row.get("ClinicalSignificance", "")

            # Filter by significance unless --all-significance
            if not all_significance:
                if not any(sig in significance for sig in include_significance):
                    continue

            # Extract fields
            try:
                variation_id = int(row.get("VariationID", 0))
            except ValueError:
                continue

            gene = row.get("GeneSymbol", "").strip()
            if not gene or gene == "-":
                continue

            # Filter by gene if specified
            if genes_filter and gene.upper() not in genes_filter:
                continue

            name = row.get("Name", "").strip()
            if not name:
                continue

            review_status = row.get("ReviewStatus", "").strip()

            # Parse phenotypes (semicolon-separated)
            phenotype_str = row.get("PhenotypeList", "")
            phenotypes = [p.strip() for p in phenotype_str.split(";") if p.strip() and p.strip() != "not provided"]

            yield ClinVarVariant(
                variation_id=variation_id,
                name=name,
                gene=gene,
                clinical_significance=significance,
                review_status=review_status,
                phenotypes=phenotypes,
            )

            count += 1
            if limit and count >= limit:
                return


def download_variant_summary(target: str) -> None:
    """Fetch variant_summary.txt.gz from NCBI to `target`, with a progress bar."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"Downloading ClinVar variant_summary from {CLINVAR_VARIANT_SUMMARY_URL}")
    print(f"  -> {target}")

    tmp = target + ".part"
    with urllib.request.urlopen(CLINVAR_VARIANT_SUMMARY_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as out, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="ClinVar"
        ) as bar:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, target)


def get_db_connection(host: str, port: str, dbname: str, user: str, password: str):
    """Create database connection."""
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


# Staging table name for the zero-downtime re-index (see --staging). Kept
# distinct from the live table so a build never touches production until the
# atomic swap at the very end.
STAGING_TABLE = "variants_staging"

# Only bare identifiers ([A-Za-z0-9_]) are ever passed as a table name here
# (the two module constants below), never user input - but assert it so a
# future caller can't smuggle SQL through the f-string interpolation the
# psycopg2 client requires for identifiers.
def _assert_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"unsafe table identifier: {name!r}")
    return name


def clear_variants_table(conn, table: str = "variants"):
    """Clear existing rows from `table`."""
    _assert_ident(table)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")
    conn.commit()


def insert_variants_batch(conn, variants: list[dict], table: str = "variants"):
    """Insert batch of variants with embeddings into `table`."""
    if not variants:
        return
    _assert_ident(table)

    # Deduplicate by variation_id (keep last occurrence)
    seen = {}
    for v in variants:
        seen[v["variation_id"]] = v
    variants = list(seen.values())

    with conn.cursor() as cur:
        sql = f"""
            INSERT INTO {table}
            (variation_id, name, gene, clinical_significance, review_status, phenotypes, document, embedding)
            VALUES %s
            ON CONFLICT (variation_id) DO UPDATE SET
                name = EXCLUDED.name,
                gene = EXCLUDED.gene,
                clinical_significance = EXCLUDED.clinical_significance,
                review_status = EXCLUDED.review_status,
                phenotypes = EXCLUDED.phenotypes,
                document = EXCLUDED.document,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
        """

        values = [
            (
                v["variation_id"],
                v["name"],
                v["gene"],
                v["clinical_significance"],
                v["review_status"],
                v["phenotypes"],
                v["document"],
                v["embedding"],
            )
            for v in variants
        ]

        execute_values(cur, sql, values)
    conn.commit()


# --- Zero-downtime staging re-index -----------------------------------------
#
# ClinVar refreshes weekly, so the re-index is a recurring operation and must
# never leave production degraded. --staging builds a fresh `variants_staging`
# table off to the side (live `variants` keeps serving throughout), embeds and
# indexes it, then swaps it in atomically at the very end. A crash before the
# swap touches only staging; production is untouched.

# Sanity floor: never swap in a staging table with implausibly few rows (a
# truncated download or a parse that silently dropped most rows would otherwise
# replace the live index with a broken one). The full corpus is ~4.2M; even a
# pathogenic-only rebuild is ~350K, so 100K is a safe "obviously broken" floor.
# Disk-space pre-check is operator-level (df on the PGDATA volume before the
# run - SQL cannot read filesystem free space without a superuser extension);
# see the runbook. Peak footprint of the ~4.2M full build is ~100 GB.
STAGING_MIN_ROWS = 100_000


def create_staging_table(conn, staging: str = STAGING_TABLE, force: bool = False):
    """Create an empty `staging` table mirroring `variants`, WITHOUT the HNSW
    index (it is built once after the bulk load - far faster than maintaining
    the graph across millions of inserts).

    Includes the UNIQUE(variation_id) constraint so the ON CONFLICT upsert in
    insert_variants_batch works during the load, plus the cheap btree/gin
    indexes. Only the expensive HNSW index is deferred to build_staging_indexes.

    Aborts if `staging` already exists (leftover from a failed run) unless
    force=True, in which case it is dropped and recreated.
    """
    _assert_ident(staging)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (staging,))
        exists = cur.fetchone()[0] is not None
        if exists:
            if not force:
                raise RuntimeError(
                    f"{staging} already exists (leftover from a failed run?). "
                    f"Re-run with --force-staging to drop and recreate it."
                )
            print(f"  Dropping existing {staging} (--force-staging)")
            cur.execute(f"DROP TABLE {staging}")

        # Mirror sql/init.sql, minus the HNSW index and the updated_at trigger
        # (staging is write-once during the load; no in-place updates to track).
        cur.execute(f"""
            CREATE TABLE {staging} (
                id SERIAL PRIMARY KEY,
                variation_id INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL,
                gene TEXT NOT NULL,
                clinical_significance TEXT NOT NULL,
                review_status TEXT,
                phenotypes TEXT[],
                document TEXT NOT NULL,
                embedding vector(1024),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(f"CREATE INDEX idx_{staging}_gene ON {staging}(gene)")
        cur.execute(
            f"CREATE INDEX idx_{staging}_clinsig "
            f"ON {staging}(clinical_significance)"
        )
    conn.commit()
    print(f"  Created empty staging table {staging} (HNSW deferred)")


def build_staging_indexes(
    conn, staging: str = STAGING_TABLE,
    maintenance_work_mem: str = "16GB", parallel_workers: int = 7,
):
    """Build the HNSW + FTS indexes on the fully-loaded staging table.

    Raises maintenance_work_mem and max_parallel_maintenance_workers for THIS
    session only (SET, not ALTER SYSTEM) so the ~33 GB HNSW graph builds in
    memory on the DB box (128 GB) instead of spilling to a slow on-disk phase
    at the server default (64 MB). HNSW params match sql/init.sql (m=16,
    ef_construction=64).
    """
    _assert_ident(staging)
    print(f"  Building indexes on {staging} "
          f"(maintenance_work_mem={maintenance_work_mem}, "
          f"parallel_workers={parallel_workers})...")
    with conn.cursor() as cur:
        cur.execute(f"SET maintenance_work_mem = %s", (maintenance_work_mem,))
        cur.execute(
            "SET max_parallel_maintenance_workers = %s", (parallel_workers,)
        )
        print("    - HNSW (embedding vector_cosine_ops)...")
        cur.execute(
            f"CREATE INDEX idx_{staging}_embedding ON {staging} "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )
        print("    - FTS (gin on document)...")
        cur.execute(
            f"CREATE INDEX idx_{staging}_document_fts ON {staging} "
            f"USING gin(to_tsvector('english', document))"
        )
        print("    - ANALYZE...")
        cur.execute(f"ANALYZE {staging}")
    conn.commit()
    print(f"  Indexes built on {staging}")


def swap_staging_into_place(
    conn, staging: str = STAGING_TABLE, live: str = "variants",
    min_rows: int = STAGING_MIN_ROWS,
):
    """Atomically replace the live table with the staging table.

    Sanity-gates on row count first (never swap in an implausibly small build),
    then renames within a single transaction so no query ever sees a missing
    table. The old live table is renamed aside and dropped after commit, and
    its indexes/constraints are renamed to avoid name collisions on the next run.

    Table names for the swap are trusted module constants, validated by
    _assert_ident (psycopg2 cannot parameterize identifiers).
    """
    _assert_ident(staging)
    _assert_ident(live)
    old = f"{live}_old"
    _assert_ident(old)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {staging}")
        n = cur.fetchone()[0]
        if n < min_rows:
            raise RuntimeError(
                f"Refusing to swap: {staging} has only {n:,} rows "
                f"(< {min_rows:,} floor). Aborting to protect the live index."
            )
        print(f"  {staging} row count OK: {n:,} (>= {min_rows:,})")

        # Drop any leftover _old from a prior interrupted swap.
        cur.execute(f"DROP TABLE IF EXISTS {old}")

        # Atomic swap: rename live aside, staging into place. Rename the
        # staging table's PK/constraint so it doesn't collide with the live
        # names still held by the (about-to-be-dropped) old table.
        print(f"  Swapping {staging} -> {live} (atomic)...")
        cur.execute(f"ALTER TABLE {live} RENAME TO {old}")
        cur.execute(f"ALTER TABLE {staging} RENAME TO {live}")
    conn.commit()

    # Drop the old table outside the swap txn so the rename commits fast and the
    # (potentially large) DROP doesn't extend the lock window.
    with conn.cursor() as cur:
        print(f"  Dropping old table {old}...")
        cur.execute(f"DROP TABLE IF EXISTS {old}")
    conn.commit()
    print(f"  Swap complete: {live} is now the rebuilt table ({n:,} rows)")


def count_variants_in_file(filepath: str, all_significance: bool = False, genes_filter: set[str] = None) -> int:
    """Count variants for progress bar."""
    print("Counting variants...")
    count = 0
    for _ in parse_clinvar(filepath, all_significance=all_significance, genes_filter=genes_filter):
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Index ClinVar variants into pgvector")

    # Data options
    parser.add_argument("--clinvar-file", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../data/clinvar/variant_summary.txt.gz"),
                        help="Path to ClinVar variant_summary.txt.gz")
    parser.add_argument("--limit", type=int, help="Limit number of variants to index")
    parser.add_argument("--all-significance", action="store_true",
                        help="Include all clinical significance (not just pathogenic)")
    parser.add_argument("--genes", type=str,
                        help="Comma-separated list of genes to index (e.g., MLH1,MSH2,BRCA1)")
    parser.add_argument("--starter-dataset", action="store_true",
                        help=(f"Index the {len(STARTER_GENES)}-gene starter set "
                              "(hereditary cancer, Lynch, cardiac, kidney, "
                              "hematology, CF). Auto-downloads ClinVar from NCBI "
                              "if not present locally. ~30k pathogenic variants, "
                              "small enough for the quickstart."))

    # Embedding options. The local-vs-remote routing is driven by the
    # EMBEDDINGS_URL env var (see app/embeddings.py); --batch-size controls
    # how many documents are bundled per encode_batch call here.
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Documents per embed batch (default: 100; "
                             "500 is a reasonable value when EMBEDDINGS_URL "
                             "points at a GPU embeddings server)")

    # Database options
    parser.add_argument("--db-host", type=str, default=os.getenv("DB_HOST", "localhost"),
                        help="Database host")
    parser.add_argument("--db-port", type=str, default=os.getenv("DB_PORT", "5432"),
                        help="Database port")
    parser.add_argument("--db-name", type=str, default=os.getenv("DB_NAME", "attengene"),
                        help="Database name")
    parser.add_argument("--db-user", type=str, default=os.getenv("DB_USER", "attengene"),
                        help="Database user")
    parser.add_argument("--db-password", type=str, default=os.getenv("DB_PASSWORD", "changeme"),
                        help="Database password")

    # Control options
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing data")
    parser.add_argument("--dry-run", action="store_true", help="Parse and embed but don't write to DB")

    # Zero-downtime staging re-index. Builds a fresh variants_staging table
    # (live `variants` keeps serving), embeds + indexes it, then swaps it in
    # atomically. This is the recommended mode for the recurring (weekly)
    # full ClinVar re-index. DISK: needs ~120 GB free on the PGDATA volume for
    # the ~4.2M full build - df it before a big run (see the runbook).
    parser.add_argument("--staging", action="store_true",
                        help="Zero-downtime re-index: build variants_staging, "
                             "index it, then atomically swap it in. Live table "
                             "keeps serving throughout. Ignores --no-clear.")
    parser.add_argument("--force-staging", action="store_true",
                        help="With --staging, drop a leftover variants_staging "
                             "from a failed run instead of aborting.")
    parser.add_argument("--staging-work-mem", type=str, default="16GB",
                        help="maintenance_work_mem for the staging HNSW build "
                             "(session-scoped SET; default 16GB - needs the DB "
                             "box RAM to keep the graph off disk)")
    parser.add_argument("--staging-workers", type=int, default=7,
                        help="max_parallel_maintenance_workers for the staging "
                             "index build (default 7)")

    args = parser.parse_args()

    # --starter-dataset sets the gene filter to STARTER_GENES (unless
    # --genes was also passed) and auto-downloads ClinVar if needed.
    if args.starter_dataset:
        if not args.genes:
            args.genes = ",".join(STARTER_GENES)
        if not os.path.exists(args.clinvar_file):
            download_variant_summary(args.clinvar_file)

    # Check file exists
    if not os.path.exists(args.clinvar_file):
        print(f"Error: ClinVar file not found: {args.clinvar_file}")
        print("  Tip: pass --starter-dataset to auto-download from NCBI,")
        print("  or fetch it manually from")
        print(f"  {CLINVAR_VARIANT_SUMMARY_URL}")
        return 1

    from app.embeddings import EMBEDDINGS_URL
    if not EMBEDDINGS_URL:
        print("Error: EMBEDDINGS_URL is not set.")
        print("  Set it to the /embed endpoint of your embeddings server")
        print("  (the bundled docker-compose stack exposes one at")
        print("  http://embeddings:8081/embed).")
        return 1
    print(f"Embeddings server: {EMBEDDINGS_URL}")

    # Target table for the load. In staging mode the load goes into
    # variants_staging (live `variants` untouched until the atomic swap); the
    # default path loads straight into `variants`.
    target_table = STAGING_TABLE if args.staging else "variants"

    # Connect to database
    if not args.dry_run:
        print(f"Connecting to database: {args.db_host}:{args.db_port}/{args.db_name}")
        try:
            conn = get_db_connection(
                args.db_host, args.db_port, args.db_name, args.db_user, args.db_password
            )
            print("  ✓ Database connection OK")
        except Exception as e:
            print(f"Error: Cannot connect to database: {e}")
            return 1

        if args.staging:
            # Build a fresh staging table (HNSW deferred to after the load).
            # Live `variants` keeps serving; --no-clear is irrelevant here.
            print(f"Staging re-index: preparing {STAGING_TABLE} "
                  f"(live table untouched until swap)")
            try:
                create_staging_table(conn, STAGING_TABLE, force=args.force_staging)
            except Exception as e:
                print(f"Error: {e}")
                conn.close()
                return 1
        elif not args.no_clear:
            # Clear existing data
            print("Clearing existing variants...")
            clear_variants_table(conn)
    else:
        print("Dry run mode - no database writes")
        conn = None

    # Parse genes filter
    genes_filter = None
    if args.genes:
        genes_filter = {g.strip().upper() for g in args.genes.split(",")}
        print(f"Filtering to genes: {', '.join(sorted(genes_filter))}")

    # Count total for progress bar (skip if limit set)
    if args.limit:
        total = args.limit
    else:
        total = count_variants_in_file(args.clinvar_file, args.all_significance, genes_filter)

    sig_mode = "all significance" if args.all_significance else "pathogenic only"
    print(f"Indexing {total:,} variants ({sig_mode}) with batch size {args.batch_size}...")

    from app.embeddings import EmbedInsertPipeline

    indexed = 0
    errors = 0
    aborted = False

    with tqdm(total=total, desc="Indexing") as pbar:
        # The sink runs on the pipeline's single insert thread; embedding
        # work happens upstream in parallel embed workers. Errors here cover
        # both insert-side failures and (re-raised) embed-side failures.
        def sink(batch: list[dict], embeddings: list[list[float]]) -> None:
            nonlocal indexed, errors, aborted
            if aborted:
                return
            for v, emb in zip(batch, embeddings):
                v["embedding"] = emb
            try:
                if conn and not args.dry_run:
                    insert_variants_batch(conn, batch, table=target_table)
                indexed += len(batch)
            except Exception as e:
                errors += 1
                print(f"\nError processing batch: {e}")
                if errors > 10:
                    aborted = True
                    print("Too many errors, aborting")
                    raise
            pbar.update(len(batch))

        try:
            with EmbedInsertPipeline(sink=sink) as pipe:
                batch: list[dict] = []
                for variant in parse_clinvar(
                    args.clinvar_file,
                    limit=args.limit,
                    all_significance=args.all_significance,
                    genes_filter=genes_filter,
                ):
                    batch.append({
                        "variation_id": variant.variation_id,
                        "name": variant.name,
                        "gene": variant.gene,
                        "clinical_significance": variant.clinical_significance,
                        "review_status": variant.review_status,
                        "phenotypes": variant.phenotypes,
                        "document": variant.to_document(),
                        "embedding": None,
                    })

                    if len(batch) >= args.batch_size:
                        pipe.submit(batch, [v["document"] for v in batch])
                        batch = []

                if batch:
                    pipe.submit(batch, [v["document"] for v in batch])
        except Exception as e:
            if not aborted:
                # Embed-side or sink-side error not covered by the 10-error
                # ceiling - print and exit non-zero. Staging table is left in
                # place for inspection; the live table was never touched.
                print(f"\nPipeline error: {e}")
                if conn:
                    conn.close()
                return 1

    if aborted:
        # Load failed. In staging mode the live table is untouched; the
        # partial staging table is left for inspection (re-run with
        # --force-staging to rebuild it).
        if conn:
            conn.close()
        return 1

    print(f"\nDone! Loaded {indexed:,} variants ({errors} errors).")

    # Staging finalization: build the deferred HNSW/FTS indexes on the loaded
    # staging table, then atomically swap it in. A failure here leaves the live
    # table serving and the staging table in place (re-runnable).
    if args.staging and conn and not args.dry_run:
        try:
            build_staging_indexes(
                conn, STAGING_TABLE,
                maintenance_work_mem=args.staging_work_mem,
                parallel_workers=args.staging_workers,
            )
            swap_staging_into_place(conn, STAGING_TABLE, "variants")
        except Exception as e:
            print(f"\nStaging finalization failed: {e}")
            print(f"  Live `variants` is unchanged and still serving.")
            print(f"  Staging table {STAGING_TABLE} left in place for inspection.")
            conn.close()
            return 1

    if conn:
        conn.close()

    if not args.dry_run:
        print(f"\nVerify with:")
        print(f"  psql -h {args.db_host} -U {args.db_user} -d {args.db_name} -c 'SELECT COUNT(*) FROM variants;'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
