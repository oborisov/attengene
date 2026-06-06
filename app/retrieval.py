"""
Deterministic vector search for ClinVar variants.

No LLM usage. Embedding model loaded once at module level.
"""

from app.db import get_connection, release_connection
from app.embeddings import encode
from app.hgvs import ParsedVariant
from app.models import VariantEvidence


def retrieve_variants_exact(parsed: ParsedVariant, k: int = 5) -> list[VariantEvidence]:
    """Exact ClinVar lookup from a parsed variant (Tier 1 of variant retrieval).

    Builds an ILIKE query on the structured tokens (gene + HGVS) rather than
    fuzzy-matching the whole sentence. High precision: an HGVS substring either
    appears in a ClinVar variant name or it does not. Returns an empty list when
    the variant is genuinely absent (clean true negative) - no cross-gene bleed,
    no similarity floor needed.

    Requires at least one concrete variant token (c./p./rsID). If only a gene is
    present, returns [] so the caller falls back to hybrid/semantic search.
    """
    if not parsed.has_variant_token:
        return []

    conds: list[str] = []
    params: list[str] = []

    # rsID is the most specific anchor when present.
    if parsed.rsid:
        conds.append("name ILIKE %s")
        params.append(f"%{parsed.rsid}%")

    # HGVS tokens: require each present token to appear in the name. AND-ing
    # c. and p. tightens precision (both must match the same variant).
    for tok in (parsed.c_hgvs, parsed.p_hgvs):
        if tok:
            conds.append("name ILIKE %s")
            params.append(f"%{tok}%")

    if not conds:
        return []

    where = " AND ".join(conds)
    # Gene narrows further when we have a confident symbol, but stays optional:
    # ClinVar names embed the gene, so a matching HGVS in the wrong gene is
    # already unlikely, and an over-eager gene guess shouldn't zero out a
    # correct HGVS hit. Apply gene only as a tie-breaker in ORDER BY - and only
    # when present (a bare "ORDER BY 0" is read by Postgres as column position
    # 0, which is invalid; omit the term entirely instead).
    order_params: list[str] = []
    if parsed.gene:
        order_by = "CASE WHEN gene = %s THEN 0 ELSE 1 END, variation_id"
        order_params.append(parsed.gene)
    else:
        order_by = "variation_id"

    sql = f"""
        SELECT variation_id, gene, name, clinical_significance, review_status,
               array_to_string(phenotypes, '; ') AS phenotypes, document
        FROM variants
        WHERE {where}
        ORDER BY {order_by}
        LIMIT %s
    """

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (*params, *order_params, k))
            rows = cur.fetchall()
    finally:
        release_connection(conn)

    results = []
    for row in rows:
        (variation_id, gene, name, sig, review, pheno, document) = row
        results.append(
            VariantEvidence(
                variation_id=variation_id,
                gene_symbol=gene,
                variant_name=name,
                clinical_significance=sig,
                review_status=review or "",
                condition_names=pheno or None,
                similarity=1.0,  # exact substring match - not a similarity score
                document=document,
            )
        )
    return results


def retrieve_variants(query: str, k: int = 5) -> list[VariantEvidence]:
    """
    Retrieve top-k variants matching the query using vector similarity.

    Args:
        query: Natural language query
        k: Number of results to return (default 5)

    Returns:
        List of VariantEvidence objects ordered by similarity (descending)
    """
    embedding = encode(query)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    variation_id,
                    gene,
                    name,
                    clinical_significance,
                    review_status,
                    array_to_string(phenotypes, '; ') AS phenotypes,
                    document,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM variants
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, k),
            )
            rows = cur.fetchall()
    finally:
        release_connection(conn)

    results = []
    for row in rows:
        (
            variation_id,
            gene,
            name,
            clinical_significance,
            review_status,
            phenotypes,
            document,
            similarity,
        ) = row

        results.append(
            VariantEvidence(
                variation_id=variation_id,
                gene_symbol=gene,
                variant_name=name,
                clinical_significance=clinical_significance,
                review_status=review_status or "",
                condition_names=phenotypes or None,
                similarity=float(similarity),
                document=document,
            )
        )

    return results


def retrieve_variants_hybrid(
    query: str, k: int = 5, alpha: float = 0.3, pool_size: int = 50,
    lexical_floor: float = 0.45,
) -> list[VariantEvidence]:
    """
    Retrieve top-k variants using hybrid search (semantic + lexical).

    Combines pgvector cosine similarity with pg_trgm trigram matching.
    Scores are min-max normalized before combining.

    For ClinVar, the lexical (trigram) leg is the meaningful signal: an HGVS
    query like c.526G>A either matches a variant name closely or it doesn't.
    The semantic leg, by contrast, returns 50 near-identical "pathogenic
    missense" neighbours, and min-max normalization then stretches that flat,
    signal-free spread across [0,1] - manufacturing fake discrimination and
    letting off-target variants surface for a variant that isn't in ClinVar
    at all (e.g. a TRIP12 query smearing onto SPAST/KRT5/AR at ~0.31).

    `lexical_floor` gates on the RAW (pre-normalization) trigram similarity:
    any candidate whose best raw lexical score is below the floor is dropped.
    A real match scores ~0.6+; trigram coincidence tops out around ~0.31. If
    nothing clears the floor, the variant is absent from ClinVar and we return
    an empty list rather than a normalized pile of noise. (Mirrors the
    GeneReviews similarity-floor fix.)

    Args:
        query: Natural language query
        k: Number of results to return (default 5)
        alpha: Weight for semantic score; (1-alpha) for lexical (default 0.3)
        pool_size: Candidates to retrieve from each method before re-ranking
        lexical_floor: Minimum raw trigram similarity for a candidate to
            survive. Below this, lexical matches are coincidental noise.

    Returns:
        List of VariantEvidence objects ordered by hybrid score (descending).
        Empty if no candidate clears the lexical floor.
    """
    embedding = encode(query)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Semantic candidates
            cur.execute(
                """
                SELECT variation_id, gene, name, clinical_significance,
                       review_status, array_to_string(phenotypes, '; '),
                       document, 1 - (embedding <=> %s::vector) AS sem_score
                FROM variants
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, pool_size),
            )
            candidates = {}
            for row in cur.fetchall():
                candidates[row[0]] = {
                    "gene": row[1], "name": row[2], "sig": row[3],
                    "review": row[4], "pheno": row[5], "doc": row[6],
                    "sem": row[7], "lex": 0.0,
                }

            # Lexical candidates (pg_trgm trigram similarity)
            cur.execute(
                """
                SELECT variation_id, gene, name, clinical_significance,
                       review_status, array_to_string(phenotypes, '; '),
                       document,
                       GREATEST(
                           similarity(name, %(q)s),
                           similarity(document, %(q)s)
                       ) AS lex_score
                FROM variants
                ORDER BY LEAST(name <-> %(q)s, document <-> %(q)s)
                LIMIT %(pool)s
                """,
                {"q": query, "pool": pool_size},
            )
            for row in cur.fetchall():
                vid = row[0]
                if vid in candidates:
                    candidates[vid]["lex"] = row[7]
                else:
                    candidates[vid] = {
                        "gene": row[1], "name": row[2], "sig": row[3],
                        "review": row[4], "pheno": row[5], "doc": row[6],
                        "sem": 0.0, "lex": row[7],
                    }
    finally:
        release_connection(conn)

    # Lexical floor on the RAW trigram score, applied before normalization.
    # A real HGVS/name match scores ~0.6+; trigram coincidence on a variant
    # that isn't in ClinVar tops out around ~0.31. Dropping sub-floor
    # candidates here means an absent-variant query returns [] instead of a
    # min-max-normalized smear of unrelated variants (which would otherwise
    # surface as authoritative-but-wrong ClinVar citations).
    candidates = {
        vid: c for vid, c in candidates.items() if c["lex"] >= lexical_floor
    }
    if not candidates:
        return []

    # Normalize scores to [0, 1]
    sem_vals = [c["sem"] for c in candidates.values()]
    lex_vals = [c["lex"] for c in candidates.values()]

    sem_min, sem_max = min(sem_vals), max(sem_vals)
    lex_min, lex_max = min(lex_vals), max(lex_vals)
    sem_range = (sem_max - sem_min) or 1.0
    lex_range = (lex_max - lex_min) or 1.0

    scored = []
    for vid, c in candidates.items():
        sem_norm = (c["sem"] - sem_min) / sem_range
        lex_norm = (c["lex"] - lex_min) / lex_range
        hybrid = alpha * sem_norm + (1 - alpha) * lex_norm
        scored.append((vid, c, hybrid))

    scored.sort(key=lambda x: x[2], reverse=True)

    results = []
    for vid, c, hybrid in scored[:k]:
        results.append(
            VariantEvidence(
                variation_id=vid,
                gene_symbol=c["gene"],
                variant_name=c["name"],
                clinical_significance=c["sig"],
                review_status=c["review"] or "",
                condition_names=c["pheno"] or None,
                similarity=float(hybrid),
                document=c["doc"],
            )
        )

    return results


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "BRCA1 pathogenic variants"
    print(f"Query: {query}\n")

    results = retrieve_variants(query, k=5)
    for i, v in enumerate(results, 1):
        print(f"[{i}] {v.gene_symbol}: {v.variant_name[:60]}...")
        print(f"    Similarity: {v.similarity:.4f}")
        print(f"    Significance: {v.clinical_significance}")
        print()
