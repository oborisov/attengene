"""
RAG retrieval for GeneReviews chunks.

Vector similarity search with optional gene/section filtering.
"""

from dataclasses import dataclass

from app.db import get_connection, release_connection
from app.embeddings import encode


@dataclass
class GeneReviewsChunk:
    """A retrieved chunk from GeneReviews."""
    id: int
    nbk_id: str
    condition_name: str
    gene_symbols: list[str]
    section_title: str
    section_path: str
    section_type: str
    chunk_text: str
    chunk_index: int
    token_estimate: int
    similarity: float  # cosine similarity score


def retrieve_genereviews(
    query: str,
    k: int = 5,
    gene_filter: list[str] = None,
    section_types: list[str] = None,
    similarity_threshold: float = 0.6,
) -> list[GeneReviewsChunk]:
    """
    Retrieve top-k GeneReviews chunks matching the query.

    Args:
        query: Natural language query
        k: Number of results to return
        gene_filter: Optional list of gene symbols to filter by
        section_types: Optional list of section types to filter by
        similarity_threshold: Minimum cosine similarity (0-1). Default 0.6:
            on-target chunks score ~0.7-0.8, while off-target "genetics
            prose" noise (e.g. an unrelated gene matching a gene query)
            tops out around 0.55, so 0.6 cleanly drops the noise.

    Returns:
        List of GeneReviewsChunk objects ordered by similarity (descending)
    """
    embedding = encode(query)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Build query with optional filters
            where_clauses = ["1 - (embedding <=> %s::vector) > %s"]
            params = [embedding, similarity_threshold]

            if gene_filter:
                where_clauses.append("gene_symbols && %s")
                params.append(gene_filter)

            if section_types:
                where_clauses.append("section_type = ANY(%s)")
                params.append(section_types)

            where_sql = " AND ".join(where_clauses)

            sql = f"""
                SELECT
                    id,
                    nbk_id,
                    condition_name,
                    gene_symbols,
                    section_title,
                    section_path,
                    section_type,
                    chunk_text,
                    chunk_index,
                    token_estimate,
                    1 - (embedding <=> %s::vector) as similarity
                FROM genereviews_chunks
                WHERE {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            # Build params: embedding for SELECT, then WHERE params, then embedding for ORDER BY, then LIMIT
            full_params = [embedding] + params + [embedding, k]

            cur.execute(sql, full_params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append(GeneReviewsChunk(
                    id=row[0],
                    nbk_id=row[1],
                    condition_name=row[2],
                    gene_symbols=row[3] or [],
                    section_title=row[4],
                    section_path=row[5],
                    section_type=row[6],
                    chunk_text=row[7],
                    chunk_index=row[8],
                    token_estimate=row[9],
                    similarity=row[10],
                ))

            return results
    finally:
        release_connection(conn)


