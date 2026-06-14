"""
Query classification and multi-database retrieval router.

Chat-focused: searches ClinVar, GeneReviews, NephroGenetics, and PubMed for
clinical questions. ClinVar hybrid search is engaged when the query carries a
gene symbol or HGVS notation (variant-level questions); it is skipped for
phenotype-only prose where the variants table only adds noise.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field

from app.citations import build_citations, format_evidence_with_citations
from app.models import Citation, VariantEvidence
from app.pubmed import search_pubmed
from app.hgvs import parse_variant
from app.retrieval import retrieve_variants_exact, retrieve_variants_hybrid
from app.retrieval_genereviews import (
    retrieve_genereviews,
    GeneReviewsChunk,
)
from app.retrieval_nephrogenetics import retrieve_nephrogenetics

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Combined results from all searched databases."""

    clinvar_evidence: list[VariantEvidence] = field(default_factory=list)
    genereviews_chunks: list[GeneReviewsChunk] = field(default_factory=list)
    nephro_results: list[dict] = field(default_factory=list)
    gnomad_data: list[dict] = field(default_factory=list)
    pubmed_abstracts: list[dict] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    prompt_context: str = ""
    sources_searched: list[str] = field(default_factory=list)
    query_type: str = ""


def _trace_enabled() -> bool:
    """Whether to render the retrieval trace. RAG_TRACE=0/false/off disables.

    Read at call time (not import) so it can be toggled per-process/in tests.
    Defaults on for the demo/pilot; flip the env var to silence it without a
    code change.
    """
    return os.getenv("RAG_TRACE", "1").strip().lower() not in ("0", "false", "off", "no")


_QUERY_TYPE_LABEL = {
    "variant": "variant lookup",
    "phenotype": "phenotype search",
    "non-clinical": "non-clinical (no retrieval)",
}


def format_retrieval_trace(result: "RetrievalResult", query: str) -> str:
    """Render a collapsible <details> "Retrieval trace" block for a query.

    Shows the query classification and, per database, whether it was searched,
    how many results came back, and the top hit's identifier/title. Databases
    that weren't searched (no signal for them) are shown as skipped, so the
    trace reflects the routing decision, not just the hits.

    Returns "" when RAG_TRACE is disabled. Pure formatting over the already-
    populated RetrievalResult - no DB access, safe to unit-test.
    """
    if not _trace_enabled():
        return ""

    label = _QUERY_TYPE_LABEL.get(result.query_type, result.query_type or "search")
    lines: list[str] = []

    # Header: classification (and genes, when this was a variant-level route).
    genes = _extract_gene_symbols(query)
    gene_note = f" ({', '.join(genes)})" if genes else ""
    lines.append(f"🔍 {label}{gene_note}")

    if result.query_type == "non-clinical":
        lines.append("· No clinical-genetics signal - databases not searched.")
        return _wrap_details(lines)

    # ClinVar - only searched for variant-level queries.
    if genes or result.clinvar_evidence:
        if result.clinvar_evidence:
            top = result.clinvar_evidence[0]
            lines.append(
                f"✓ ClinVar — {_count(len(result.clinvar_evidence), 'variant')} "
                f"(top: Variation {top.variation_id}, {top.gene_symbol}, "
                f"{top.clinical_significance})"
            )
        else:
            lines.append("· ClinVar — no matching variants")
    else:
        lines.append("· ClinVar — skipped (phenotype-only query)")

    # GeneReviews - always searched.
    if result.genereviews_chunks:
        top = result.genereviews_chunks[0]
        lines.append(
            f"✓ GeneReviews — {_count(len(result.genereviews_chunks), 'chapter')} "
            f"(top: {top.condition_name} [{top.nbk_id}])"
        )
    else:
        lines.append("· GeneReviews — no matching chapters")

    # NephroGenetics - only when renal terms are present.
    if _has_nephro_terms(query):
        if result.nephro_results:
            top = result.nephro_results[0]
            lines.append(
                f"✓ NephroGenetics — {_count(len(result.nephro_results), 'entry')} "
                f"(top: {top.get('gene', '?')} - {top.get('title', '?')})"
            )
        else:
            lines.append("· NephroGenetics — no matching entries")
    else:
        lines.append("· NephroGenetics — skipped (no renal terms)")

    # PubMed - only when a gene symbol was parsed.
    if genes:
        if result.pubmed_abstracts:
            top = result.pubmed_abstracts[0]
            lines.append(
                f"✓ PubMed — {_count(len(result.pubmed_abstracts), 'abstract')} "
                f"(top: PMID {top.get('pmid', '?')})"
            )
        else:
            lines.append("· PubMed — no matching abstracts")
    else:
        lines.append("· PubMed — skipped (no gene symbol)")

    return _wrap_details(lines)


def _count(n: int, noun: str) -> str:
    """'1 variant' / '3 variants', '1 entry' / '3 entries'."""
    if n == 1:
        return f"{n} {noun}"
    plural = noun[:-1] + "ies" if noun.endswith("y") else noun + "s"
    return f"{n} {plural}"


def _wrap_details(lines: list[str]) -> str:
    """Wrap trace lines in a collapsed <details> block, OWUI/markdown-safe."""
    body = "\n".join(lines)
    return (
        "<details>\n<summary>🔎 Retrieval trace</summary>\n\n"
        f"```\n{body}\n```\n\n</details>\n\n"
    )


# Kidney/renal terms for NephroGenetics routing
_NEPHRO_TERMS = [
    "kidney", "renal", "nephro", "nephrotic", "nephritic", "glomerul",
    "tubul", "polycystic", "pkd", "alport", "fabry", "iga nephropathy",
    "fsgs", "cakut", "ckd", "dialysis", "transplant",
]

# Pattern to extract gene symbols from query.
# Uppercase tokens are taken as gene symbols directly (gene symbols are
# uppercase by HGNC convention - "BRCA1", "COL4A3").
_GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,9})\b')

# Lowercase/mixed-case gene tokens are only accepted when a nearby cue marks
# them as a gene ("muc1 gene", "gene KMT2D", "variant in brca1"). Matching
# lowercase unconditionally would treat every prose word ("what", "kidney",
# "causes") as a candidate and wrongly flip phenotype queries to the
# variant route. The cue keeps recall (clinicians type lowercase) without
# that flood. The symbol itself is the same shape as _GENE_PATTERN but
# case-insensitive; we uppercase the capture to the canonical form.
_GENE_TOKEN = r'[A-Za-z][A-Za-z0-9]{1,9}'
_CUED_GENE_PATTERN = re.compile(
    rf'\b(?:(?:gene|variant|mutation|allele)s?\s+(?:in\s+|of\s+)?({_GENE_TOKEN})'
    rf'|({_GENE_TOKEN})\s+gene)\b',
    re.IGNORECASE,
)

# Common English words that look like gene symbols
_NOT_GENES = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM", "LET", "SAY",
    "SHE", "TOO", "USE", "DNA", "RNA", "VUS", "WHAT", "WHICH", "WITH",
    "FROM", "HAVE", "THIS", "THAT", "THEY", "BEEN", "RISK", "TYPE", "GENE",
    "DOES", "MOST", "RARE", "ALSO", "MORE", "THAN", "INTO", "SOME",
}


def _extract_gene_symbols(query: str) -> list[str]:
    """Extract likely gene symbols from query text.

    Two sources, both normalized to uppercase canonical form:
      - bare uppercase tokens ("BRCA1"), as before;
      - cued lowercase/mixed-case tokens ("muc1 gene", "gene kmt2d",
        "variant in brca1"), so clinicians typing lowercase still get the
        variant-level route. The cue requirement prevents prose words from
        being mistaken for genes.

    Stopwords (THE, AND, GENE, ...) are filtered after upper-casing, so the
    cue word itself ("gene") never survives as a candidate. Order is
    preserved and duplicates removed.
    """
    candidates = list(_GENE_PATTERN.findall(query))
    for m in _CUED_GENE_PATTERN.finditer(query):
        # Exactly one of the two alternation groups matches per cue.
        token = m.group(1) or m.group(2)
        if token:
            candidates.append(token.upper())

    seen: set[str] = set()
    result: list[str] = []
    for g in candidates:
        g = g.upper()
        if g in _NOT_GENES or len(g) < 2 or g in seen:
            continue
        seen.add(g)
        result.append(g)
    return result


def _has_nephro_terms(query: str) -> bool:
    """Check if query mentions kidney/renal topics."""
    query_lower = query.lower()
    return any(term in query_lower for term in _NEPHRO_TERMS)


# Cheap topical signal: HGVS notation, ACMG-class words, common clinical
# nouns. Used to skip pgvector entirely on greetings like "hi" / "thanks"
# where any retrieval result is noise.
_CLINICAL_SIGNAL_TERMS = [
    "gene", "variant", "mutation", "allele", "phenotype", "syndrome",
    "disease", "disorder", "inherit", "heredit", "pathogen", "benign",
    "clinvar", "omim", "acmg", "vus", "genotype", "carrier", "homozyg",
    "heterozyg", "autosomal", "x-linked", "recessive", "dominant",
    "diagnos", "screen", "panel", "exome", "genome", "sequenc",
]
_HGVS_PATTERN = re.compile(r"\b[cpgmn]\.[0-9]")  # c.123 / p.Arg / etc.


def _has_clinical_signal(query: str) -> bool:
    """True if the query plausibly relates to clinical genetics."""
    if _HGVS_PATTERN.search(query):
        return True
    if _extract_gene_symbols(query):
        return True
    query_lower = query.lower()
    if any(term in query_lower for term in _CLINICAL_SIGNAL_TERMS):
        return True
    return _has_nephro_terms(query)


def route_and_retrieve(query: str, k: int = 10) -> RetrievalResult:
    """
    Search ClinVar, GeneReviews, NephroGenetics, and PubMed for clinical context.

    ClinVar hybrid search runs only for variant-level queries - those that
    name a gene symbol or carry HGVS notation. Phenotype-only prose skips
    ClinVar, because the variants table only contributes noise there and the
    nearest-neighbour match would leak an off-target variant into citations.
    """
    result = RetrievalResult()
    t0 = time.perf_counter()

    # Skip retrieval entirely for queries with no clinical-genetics signal
    # (greetings, smalltalk). pgvector returns the nearest neighbour for
    # any input, which on "hi" surfaces a random author affiliation and
    # confuses the LLM into citing it. With empty context, the LLM falls
    # back to a plain greeting.
    if not _has_clinical_signal(query):
        logger.info("Skipping retrieval (no clinical signal): %r", query)
        result.query_type = "non-clinical"
        return result

    gene_symbols = _extract_gene_symbols(query)
    has_hgvs = bool(_HGVS_PATTERN.search(query))
    result.query_type = "variant" if (gene_symbols or has_hgvs) else "phenotype"

    # Search ClinVar for variant-level queries, two tiers:
    #   Tier 1 (exact): parse gene + HGVS tokens out of the query and do an
    #     exact ILIKE lookup. High precision, clean true negatives, immune to
    #     the sentence-dilution problem that defeats fuzzy matching on a
    #     conversational query ("what about the variant c.526G>A ... in ALPL").
    #   Tier 2 (hybrid): fall back to lexical+semantic search only when Tier 1
    #     finds nothing - e.g. phenotype-ish variant queries with no parseable
    #     HGVS token ("what variants are known in BRCA1").
    if gene_symbols or has_hgvs:
        try:
            t1 = time.perf_counter()
            parsed = parse_variant(query)
            result.clinvar_evidence = retrieve_variants_exact(parsed, k=k)
            tier = "exact"
            if not result.clinvar_evidence:
                result.clinvar_evidence = retrieve_variants_hybrid(query, k=k)
                tier = "hybrid"
            if result.clinvar_evidence:
                result.sources_searched.append("ClinVar")
            logger.info("ClinVar search (%s): %.1fs, %d variants",
                         tier, time.perf_counter() - t1, len(result.clinvar_evidence))
        except Exception:
            logger.exception("ClinVar search failed")

    # Always search GeneReviews (primary source for clinical questions).
    #
    # When the query names gene symbols, anchor retrieval on them via
    # gene_filter. Pure semantic search drifts badly here: a query about a
    # gene that isn't in the corpus (e.g. WFS1) still returns the densest
    # "genetics prose" chunk - an unrelated gene like CFTR scoring ~0.56 -
    # which then leaks into the citations even when the answer correctly
    # says "no evidence". Filtering by gene makes off-target matches
    # impossible: if the named gene isn't indexed, we get an empty result
    # and the LLM reports no evidence with no misleading reference. We fall
    # back to an unfiltered semantic search only when no gene symbol is
    # present (e.g. phenotype-only questions).
    try:
        t1 = time.perf_counter()
        result.genereviews_chunks = retrieve_genereviews(
            query, k=k, gene_filter=gene_symbols or None
        )
        if not result.genereviews_chunks and gene_symbols:
            # Named gene not in the GeneReviews corpus - do NOT fall back to
            # an unfiltered search, which would surface an off-target chunk.
            logger.info("GeneReviews: no chunks for gene(s) %s", gene_symbols)
        if result.genereviews_chunks:
            result.sources_searched.append("GeneReviews")
        logger.info("GeneReviews search: %.1fs, %d chunks",
                     time.perf_counter() - t1, len(result.genereviews_chunks))
    except Exception:
        logger.exception("GeneReviews search failed")

    # Search NephroGenetics if kidney/renal terms present
    if _has_nephro_terms(query):
        try:
            t1 = time.perf_counter()
            result.nephro_results = retrieve_nephrogenetics(query, k=k)
            if result.nephro_results:
                result.sources_searched.append("NephroGenetics")
            logger.info("NephroGenetics search: %.1fs", time.perf_counter() - t1)
        except Exception:
            logger.exception("NephroGenetics search failed")

    # Search PubMed using gene symbols from query (computed above)
    if gene_symbols:
        try:
            t1 = time.perf_counter()
            result.pubmed_abstracts = search_pubmed(gene_symbols[0], max_results=2)
            if result.pubmed_abstracts:
                result.sources_searched.append("PubMed")
            logger.info("PubMed search: %.1fs", time.perf_counter() - t1)
        except Exception:
            logger.exception("PubMed search failed")

    logger.info("Total retrieval: %.1fs (sources: %s)",
                time.perf_counter() - t0, result.sources_searched)

    # Build citations and format evidence for the LLM
    result.citations = build_citations(
        clinvar_evidence=result.clinvar_evidence,
        genereviews_chunks=result.genereviews_chunks,
        nephro_results=result.nephro_results,
        gnomad_data=[],
        pubmed_abstracts=result.pubmed_abstracts,
    )
    result.prompt_context = format_evidence_with_citations(
        citations=result.citations,
        clinvar_evidence=result.clinvar_evidence,
        genereviews_chunks=result.genereviews_chunks,
        nephro_results=result.nephro_results,
        gnomad_data=[],
        pubmed_abstracts=result.pubmed_abstracts,
    )

    return result
