#!/usr/bin/env python3
"""
Index the NephroGenetics IKD gene table into pgvector.

Creates searchable documents from the 404 genes with their kidney
and extrarenal manifestations.

Usage:
    python scripts/index_nephrogenetics.py
"""

import os
import sys
import argparse

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

# Database configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attengene")
DB_USER = os.getenv("DB_USER", "attengene")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def create_table(conn):
    """Create the nephrogenetics table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nephrogenetics (
                id SERIAL PRIMARY KEY,
                gene_symbol TEXT NOT NULL,
                omim_phenotype_id TEXT,
                omim_gene_id TEXT,
                title TEXT,
                inheritance TEXT,
                panel_categories TEXT,
                kidney_manifestations TEXT,
                extrarenal_manifestations JSONB,
                document TEXT NOT NULL,
                embedding vector(1024),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(gene_symbol, omim_phenotype_id)
            );

            CREATE INDEX IF NOT EXISTS idx_nephrogenetics_gene ON nephrogenetics(gene_symbol);
            CREATE INDEX IF NOT EXISTS idx_nephrogenetics_embedding ON nephrogenetics
                USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()


def load_excel(filepath: str) -> pd.DataFrame:
    """Load the IKD gene table."""
    df = pd.read_excel(filepath, header=4)
    print(f"Loaded {len(df)} genes with {len(df.columns)} columns")
    return df


def parse_manifestations(row: pd.Series) -> dict:
    """Extract organ system manifestations from a row."""
    # Map column indices to organ systems
    organ_systems = {
        "height": 10,
        "weight": 11,
        "face": 14,
        "ears": 15,
        "eyes": 16,
        "nose": 17,
        "mouth": 18,
        "teeth": 19,
        "neck": 20,
        "heart": 21,
        "vascular": 22,
        "respiratory": (24, 27),  # nasopharynx to lung
        "liver": 34,
        "pancreas": 35,
        "gastrointestinal": 38,
        "genitourinary": (40, 45),  # external to bladder
        "skull": 47,
        "spine": 48,
        "limbs": 50,
        "hands": 51,
        "feet": 52,
        "skin": 53,
        "nails": 56,
        "hair": 57,
        "muscle": 58,
        "cns": 60,
        "peripheral_nervous": 61,
        "behavioral": 62,
        "metabolic": 64,
        "endocrine": 65,
        "hematology": 66,
        "immunology": 67,
        "neoplasia": 68,
    }

    manifestations = {}
    cols = row.index.tolist()

    for system, idx in organ_systems.items():
        if isinstance(idx, tuple):
            # Range of columns
            values = []
            for i in range(idx[0], idx[1] + 1):
                if i < len(cols):
                    val = row.iloc[i]
                    if pd.notna(val) and str(val).strip():
                        values.append(str(val).strip())
            if values:
                manifestations[system] = "; ".join(values)
        else:
            if idx < len(cols):
                val = row.iloc[idx]
                if pd.notna(val) and str(val).strip():
                    manifestations[system] = str(val).strip()

    return manifestations


def create_document(row: pd.Series, manifestations: dict) -> str:
    """Create a searchable document from a gene row."""
    gene = str(row.iloc[0]) if pd.notna(row.iloc[0]) else "Unknown"
    title = str(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
    inheritance = str(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
    kidneys = str(row.iloc[9]) if pd.notna(row.iloc[9]) else ""

    doc_parts = [
        f"Gene: {gene}",
        f"Condition: {title}" if title else "",
        f"Inheritance: {inheritance}" if inheritance else "",
        f"Kidney manifestations: {kidneys}" if kidneys else "",
    ]

    if manifestations:
        extrarenal = []
        for system, value in manifestations.items():
            extrarenal.append(f"{system.replace('_', ' ').title()}: {value}")
        if extrarenal:
            doc_parts.append("Extrarenal manifestations: " + "; ".join(extrarenal))

    return " ".join([p for p in doc_parts if p])


def main():
    parser = argparse.ArgumentParser(description="Index NephroGenetics IKD table")
    parser.add_argument(
        "--excel-file",
        default="data/nephrogenetics/Table2_20240605.xlsx",
        help="Path to the Excel file",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    # Load data
    df = load_excel(args.excel_file)

    from app.embeddings import EMBEDDINGS_URL
    if not EMBEDDINGS_URL:
        print("Error: EMBEDDINGS_URL is not set. Point it at your "
              "embeddings server's /embed endpoint.")
        return 1
    print(f"Embeddings server: {EMBEDDINGS_URL}")

    # Connect to database
    print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    conn = get_connection()

    # Create table
    create_table(conn)

    # Clear existing data
    if not args.no_clear:
        print("Clearing existing data...")
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE nephrogenetics RESTART IDENTITY")
        conn.commit()

    # Process genes
    print(f"Indexing {len(df)} genes...")
    from app.embeddings import EmbedInsertPipeline

    indexed = 0

    INSERT_SQL = """
        INSERT INTO nephrogenetics
        (gene_symbol, omim_phenotype_id, omim_gene_id, title, inheritance,
         panel_categories, kidney_manifestations, extrarenal_manifestations,
         document, embedding)
        VALUES %s
        ON CONFLICT (gene_symbol, omim_phenotype_id) DO UPDATE SET
            title = EXCLUDED.title,
            inheritance = EXCLUDED.inheritance,
            panel_categories = EXCLUDED.panel_categories,
            kidney_manifestations = EXCLUDED.kidney_manifestations,
            extrarenal_manifestations = EXCLUDED.extrarenal_manifestations,
            document = EXCLUDED.document,
            embedding = EXCLUDED.embedding
    """

    def sink(batch: list[dict], embeddings: list[list[float]]) -> None:
        nonlocal indexed
        for b, emb in zip(batch, embeddings):
            b["embedding"] = emb
        with conn.cursor() as cur:
            values = [
                (
                    b["gene_symbol"],
                    b["omim_phenotype_id"],
                    b["omim_gene_id"],
                    b["title"],
                    b["inheritance"],
                    b["panel_categories"],
                    b["kidney_manifestations"],
                    psycopg2.extras.Json(b["extrarenal_manifestations"]),
                    b["document"],
                    b["embedding"],
                )
                for b in batch
            ]
            execute_values(cur, INSERT_SQL, values)
        conn.commit()
        indexed += len(batch)

    def dedupe(batch: list[dict]) -> list[dict]:
        seen: dict[tuple, dict] = {}
        for b in batch:
            seen[(b["gene_symbol"], b["omim_phenotype_id"])] = b
        return list(seen.values())

    with EmbedInsertPipeline(sink=sink) as pipe:
        batch: list[dict] = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            gene = str(row.iloc[0]) if pd.notna(row.iloc[0]) else None
            if not gene or gene == "nan":
                continue

            omim_pheno = str(row.iloc[4]) if pd.notna(row.iloc[4]) else None
            omim_gene = str(row.iloc[5]) if pd.notna(row.iloc[5]) else None
            title = str(row.iloc[6]) if pd.notna(row.iloc[6]) else None
            inheritance = str(row.iloc[7]) if pd.notna(row.iloc[7]) else None
            panels = str(row.iloc[1]) if pd.notna(row.iloc[1]) else None
            kidneys = str(row.iloc[9]) if pd.notna(row.iloc[9]) else None

            manifestations = parse_manifestations(row)
            document = create_document(row, manifestations)

            batch.append({
                "gene_symbol": gene,
                "omim_phenotype_id": omim_pheno,
                "omim_gene_id": omim_gene,
                "title": title,
                "inheritance": inheritance,
                "panel_categories": panels,
                "kidney_manifestations": kidneys,
                "extrarenal_manifestations": manifestations,
                "document": document,
            })

            if len(batch) >= args.batch_size:
                deduped = dedupe(batch)
                pipe.submit(deduped, [b["document"] for b in deduped])
                batch = []

        if batch:
            deduped = dedupe(batch)
            pipe.submit(deduped, [b["document"] for b in deduped])

    conn.close()
    print(f"\nDone! Indexed {indexed} gene entries.")


if __name__ == "__main__":
    sys.exit(main() or 0)
