"""
RAG retrieval for NephroGenetics IKD gene table.
"""

from app.db import get_connection, release_connection
from app.embeddings import encode


def retrieve_nephrogenetics(
    query: str,
    k: int = 10,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """
    Retrieve top-k IKD genes matching the query.

    Returns list of dicts with gene info.
    """
    embedding = encode(query)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    gene_symbol,
                    title,
                    inheritance,
                    kidney_manifestations,
                    extrarenal_manifestations,
                    omim_phenotype_id,
                    1 - (embedding <=> %s::vector) as similarity
                FROM nephrogenetics
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, similarity_threshold, embedding, k),
            )
            rows = cur.fetchall()

            return [
                {
                    "gene": row[0],
                    "title": row[1],
                    "inheritance": row[2],
                    "kidney": row[3],
                    "extrarenal": row[4] or {},
                    "omim": row[5],
                    "similarity": row[6],
                }
                for row in rows
            ]
    finally:
        release_connection(conn)
