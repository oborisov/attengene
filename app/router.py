"""
Query classification and multi-database retrieval router.

Chat-focused: searches ClinVar, GeneReviews, NephroGenetics, and PubMed for
clinical questions. ClinVar hybrid search is engaged when the query carries a
gene symbol or HGVS notation (variant-level questions); it is skipped for
phenotype-only prose where the variants table only adds noise.
"""

import logging
import re
import time
from dataclasses import dataclass, field

from app.citations import build_citations, format_evidence_with_citations
from app.models import Citation, VariantEvidence
from app.pubmed import search_pubmed
from app.retrieval import retrieve_variants_hybrid
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


# Kidney/renal terms for NephroGenetics routing
_NEPHRO_TERMS = [
    "kidney", "renal", "nephro", "nephrotic", "nephritic", "glomerul",
    "tubul", "polycystic", "pkd", "alport", "fabry", "iga nephropathy",
    "fsgs", "cakut", "ckd", "dialysis", "transplant",
]

# Pattern to extract gene symbols from query
_GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,9})\b')

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
    """Extract likely gene symbols from query text."""
    candidates = _GENE_PATTERN.findall(query)
    return [g for g in candidates if g not in _NOT_GENES and len(g) >= 2]


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

    # Search ClinVar (hybrid lexical+semantic) for variant-level queries.
    # The lexical (pg_trgm) leg matches exact HGVS notation like c.526G>A,
    # which pure semantic search misses; the gene/HGVS gate keeps the
    # variants table out of phenotype-only prose where it adds only noise.
    if gene_symbols or has_hgvs:
        try:
            t1 = time.perf_counter()
            result.clinvar_evidence = retrieve_variants_hybrid(query, k=k)
            if result.clinvar_evidence:
                result.sources_searched.append("ClinVar")
            logger.info("ClinVar search: %.1fs, %d variants",
                         time.perf_counter() - t1, len(result.clinvar_evidence))
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
