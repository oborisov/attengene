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

    # Custom database host
    python scripts/index_clinvar.py --db-host YOUR_DB_HOST
"""

import gzip
import csv
import os
import argparse
import sys
import urllib.request
from typing import Generator
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from dotenv import load_dotenv

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


def clear_variants_table(conn):
    """Clear existing variants."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE variants RESTART IDENTITY")
    conn.commit()


def insert_variants_batch(conn, variants: list[dict]):
    """Insert batch of variants with embeddings."""
    if not variants:
        return

    # Deduplicate by variation_id (keep last occurrence)
    seen = {}
    for v in variants:
        seen[v["variation_id"]] = v
    variants = list(seen.values())

    with conn.cursor() as cur:
        sql = """
            INSERT INTO variants
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

        # Clear existing data
        if not args.no_clear:
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
                    insert_variants_batch(conn, batch)
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
                # ceiling - print and exit non-zero.
                print(f"\nPipeline error: {e}")
                if conn:
                    conn.close()
                return 1

    if conn:
        conn.close()

    if aborted:
        return 1

    print(f"\nDone! Indexed {indexed:,} variants ({errors} errors).")

    if not args.dry_run:
        print(f"\nVerify with:")
        print(f"  psql -h {args.db_host} -U {args.db_user} -d {args.db_name} -c 'SELECT COUNT(*) FROM variants;'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
