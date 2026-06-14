"""
Deterministic vector search for ClinVar variants.

No LLM usage. Embedding model loaded once at module level.
"""

from app.db import get_connection, release_connection
from app.embeddings import encode
from app.hgvs import ParsedVariant, hgvs_match_forms
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
    #
    # The coding token may have more than one valid ClinVar surface form
    # (ClinVar stores deletions both with and without trailing bases, e.g.
    # c.730_731delAG vs c.1521_1523del); OR its forms so we match whichever is
    # stored. The p. token and rsID have a single form.
    if parsed.c_hgvs:
        forms = hgvs_match_forms(parsed.c_hgvs)
        conds.append("(" + " OR ".join(["name ILIKE %s"] * len(forms)) + ")")
        params.extend(f"%{f}%" for f in forms)
    if parsed.p_hgvs:
        conds.append("name ILIKE %s")
        params.append(f"%{parsed.p_hgvs}%")

    if not conds:
        return []

    # Gene narrows to the gene in scope when we have a confident symbol. This
    # is a FILTER, not just an ORDER BY tie-breaker: HGVS coordinates are not
    # unique across the genome - common ones (e.g. c.203C>T, c.526G>A) exist in
    # multiple genes, so an HGVS-only match pulls in same-coordinate variants
    # from unrelated genes (ALPL query -> GP1BB/PNPT1 ride along). Sorting alone
    # still returns them under LIMIT k, where they surface as authoritative-but-
    # wrong-gene ClinVar citations. Filtering makes that impossible.
    #
    # When the gene is filtered to empty, that is the correct answer: the variant
    # is not in THIS gene. We return [] (a clean true negative) rather than
    # leaking a same-coordinate hit from another gene; the caller then falls back
    # to prose/hybrid and reports absence.
    #
    # When no gene was parsed (e.g. a gene-less follow-up "what about c.526G>A"),
    # we cannot filter and keep the old ungated behavior - ordering by
    # variation_id - so a correct hit is never zeroed out. (Threading the
    # last-known gene from conversation history into retrieval is a separate,
    # larger change.)
    if parsed.gene:
        conds.append("gene = %s")
        params.append(parsed.gene)

    where = " AND ".join(conds)
    sql = f"""
        SELECT variation_id, gene, name, clinical_significance, review_status,
               array_to_string(phenotypes, '; ') AS phenotypes, document
        FROM variants
        WHERE {where}
        ORDER BY variation_id
        LIMIT %s
    """

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (*params, k))
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


# ClinVar review-status quality tiers (the "gold stars"), best first. Used to
# surface the most authoritative variants when a query names only a gene.
_REVIEW_STATUS_RANK = {
    "practice guideline": 0,
    "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 4,
    "criteria provided, conflicting interpretations": 4,
    "no assertion criteria provided": 5,
    "no assertion for the individual variant": 6,
    "no classification provided": 7,
}
_REVIEW_STATUS_RANK_DEFAULT = 8


def retrieve_variants_by_gene(gene: str, k: int = 5) -> list[VariantEvidence]:
    """Exact gene-column lookup for a bare gene query (e.g. "BRCA1").

    A query that names a gene but carries no HGVS token has an exact match
    sitting in the `gene` column, but neither the HGVS-exact tier (no variant
    token) nor the hybrid tier finds it: a short gene symbol trigram-matches a
    full variant *name* far below the lexical floor (e.g. "BRCA1" vs
    "NM_007294.4(BRCA1):c.190T>G" scores ~0.23 < 0.45), so hybrid drops every
    candidate and returns nothing. This tier closes that gap with a direct
    `gene = %s` filter.

    Ordered by ClinVar review status (expert-panel/multi-submitter first) so
    the k returned are the most authoritative variants for the gene, not an
    arbitrary slice. Empty list when the gene is absent from ClinVar.
    """
    if not gene:
        return []

    # Rank review_status in SQL via a CASE so ordering happens in the DB. The
    # rank ints are trusted constants (inlined); only the status strings are
    # parameterized.
    when_clauses = "\n".join(
        f"WHEN review_status = %s THEN {rank}"
        for status, rank in _REVIEW_STATUS_RANK.items()
    )
    status_params = list(_REVIEW_STATUS_RANK.keys())

    sql = f"""
        SELECT variation_id, gene, name, clinical_significance, review_status,
               array_to_string(phenotypes, '; ') AS phenotypes, document
        FROM variants
        WHERE gene = %s
        ORDER BY
            CASE
                {when_clauses}
                ELSE {_REVIEW_STATUS_RANK_DEFAULT}
            END,
            variation_id
        LIMIT %s
    """

    # Params must be in the SQL's textual placeholder order: WHERE gene first,
    # then the CASE status strings, then LIMIT.
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (gene, *status_params, k))
            rows = cur.fetchall()
    finally:
        release_connection(conn)

    results = []
    for row in rows:
        (variation_id, gene_sym, name, sig, review, pheno, document) = row
        results.append(
            VariantEvidence(
                variation_id=variation_id,
                gene_symbol=gene_sym,
                variant_name=name,
                clinical_significance=sig,
                review_status=review or "",
                condition_names=pheno or None,
                similarity=1.0,  # exact gene match - not a similarity score
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
