"""
Citation building, evidence formatting, and post-processing.

Assigns sequential [N] numbers to all retrieved evidence, formats context
for the LLM prompt, and post-processes responses to add clickable links
and a References section.
"""

import json
import re

from app.models import Citation, CitationSource, VariantEvidence
from app.retrieval_genereviews import GeneReviewsChunk

# The heading of a References section the LLM wrote itself (it is prompted to
# cite, and some models append their own "References:" list). Matches an optional
# markdown rule, optional "**"/"#" decoration, the word References, and a
# trailing ':' or newline. Shared with the streaming path so it can stop
# forwarding the model's own block before its first character is shipped.
REFERENCES_HEADING = re.compile(
    r"\n+(?:-{3,}\s*\n+)?"        # optional horizontal rule
    r"(?:\*{0,2}|#{1,6}\s*)"      # optional bold / heading markers
    r"References\b\s*:?\**",      # the heading word + optional colon/bold
    re.IGNORECASE,
)

# Same heading plus everything after it to end-of-text: the full block to strip
# from a complete (non-streamed) response before we append our canonical one,
# so the answer never ends up with two References sections.
_LLM_REFERENCES_BLOCK = re.compile(
    REFERENCES_HEADING.pattern + r".*$",
    re.IGNORECASE | re.DOTALL,
)

# Phrases the model uses to signal the queried item is absent from evidence.
# When present, the appended References block is suppressed (the listed
# sources are off-target neighbours, not support for the answer).
#
# German clinicians are the primary user base and the LLM answers German
# prompts in German, so the not-found phrasings must be matched in German too -
# otherwise the suppression silently fails and stray off-target citations
# return on a German "no evidence" answer.
_NO_EVIDENCE_PATTERN = re.compile(
    # English
    r"not (?:mentioned|described|listed|found|present|available) in "
    r"the (?:retrieved |provided )?evidence"
    r"|does not contain sufficient information"
    r"|no (?:relevant |matching )?(?:evidence|variants?) (?:was|were) (?:retrieved|found)"
    # German: "nicht in den (bereitgestellten/abgerufenen) Belegen/Nachweisen
    # (erwähnt/enthalten/aufgeführt)"
    r"|nicht in den (?:bereitgestellten |abgerufenen )?"
    r"(?:belegen|nachweisen|quellen|daten)"
    # German: "keine (relevanten/passenden) Belege/Nachweise/Varianten
    # (gefunden/vorhanden)"
    r"|keine (?:relevanten |passenden )?"
    r"(?:belege|nachweise|hinweise|varianten?|informationen) "
    r"(?:gefunden|vorhanden|verfügbar)"
    # German: "(wird/ist) nicht erwähnt", "nicht aufgeführt", "nicht enthalten"
    r"|nicht (?:erwähnt|aufgeführt|enthalten|aufgelistet)"
    # German: "enthält nicht genügend/keine Informationen"
    r"|enthält (?:nicht genügend|keine) information",
    re.IGNORECASE,
)


def build_citations(
    clinvar_evidence: list[VariantEvidence] | None = None,
    genereviews_chunks: list[GeneReviewsChunk] | None = None,
    nephro_results: list[dict] | None = None,
    gnomad_data: list[dict] | None = None,
    pubmed_abstracts: list[dict] | None = None,
) -> list[Citation]:
    """
    Build sequentially numbered citations from all retrieval sources.

    Order: ClinVar (primary), GeneReviews, NephroGenetics, gnomAD, PubMed.
    """
    citations: list[Citation] = []
    n = 1

    # ClinVar
    for v in clinvar_evidence or []:
        sig = v.clinical_significance or "Unknown significance"
        title = f"ClinVar:{v.variation_id} - {v.gene_symbol} {v.variant_name}, {sig}"
        url = f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{v.variation_id}/"
        detail = v.document
        citations.append(Citation(
            number=n, source=CitationSource.CLINVAR,
            title=title, url=url, detail=detail,
        ))
        n += 1

    # GeneReviews - deduplicate by nbk_id
    seen_nbk = set()
    for chunk in genereviews_chunks or []:
        if chunk.nbk_id in seen_nbk:
            continue
        seen_nbk.add(chunk.nbk_id)
        genes = ", ".join(chunk.gene_symbols) if chunk.gene_symbols else ""
        title_parts = [f"GeneReviews: {chunk.condition_name}"]
        if genes:
            title_parts.append(f"({genes})")
        title_parts.append(f"[{chunk.nbk_id}]")
        title = " ".join(title_parts)
        url = f"https://www.ncbi.nlm.nih.gov/books/{chunk.nbk_id}/"
        citations.append(Citation(
            number=n, source=CitationSource.GENEREVIEWS,
            title=title, url=url,
        ))
        n += 1

    # NephroGenetics
    for r in nephro_results or []:
        omim_id = r.get("omim")
        title = f"NephroGenetics: {r['gene']} - {r['title']}"
        url = f"https://omim.org/entry/{omim_id}" if omim_id else ""
        citations.append(Citation(
            number=n, source=CitationSource.NEPHROGENETICS,
            title=title, url=url,
        ))
        n += 1

    # gnomAD
    for g in gnomad_data or []:
        af = g.get("allele_frequency", "N/A")
        title = f"gnomAD: {g['variant_id']}, AF={af}"
        url = g.get("url", "")
        citations.append(Citation(
            number=n, source=CitationSource.GNOMAD,
            title=title, url=url,
        ))
        n += 1

    # PubMed
    for p in pubmed_abstracts or []:
        title = f"PMID:{p['pmid']} - {p['title']}"
        url = p.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/")
        citations.append(Citation(
            number=n, source=CitationSource.PUBMED,
            title=title, url=url,
            detail=p.get("abstract", ""),
        ))
        n += 1

    return citations


def format_evidence_with_citations(
    citations: list[Citation],
    clinvar_evidence: list[VariantEvidence] | None = None,
    genereviews_chunks: list[GeneReviewsChunk] | None = None,
    nephro_results: list[dict] | None = None,
    gnomad_data: list[dict] | None = None,
    pubmed_abstracts: list[dict] | None = None,
) -> str:
    """
    Format evidence context for the LLM prompt with [N] citation markers.

    Each piece of evidence is tagged with its citation number so the LLM
    can reference it naturally as [1], [2], etc.
    """
    if not citations:
        return ""

    parts: list[str] = []

    # Build lookup from citation number to citation
    by_number = {c.number: c for c in citations}

    # ClinVar evidence
    for v in clinvar_evidence or []:
        c = _find_citation(citations, CitationSource.CLINVAR, str(v.variation_id))
        if not c:
            continue
        entry = f"[{c.number}] ClinVar:{v.variation_id} - {v.gene_symbol} {v.variant_name} ({v.clinical_significance}, {v.review_status})"
        if v.condition_names:
            entry += f"\nConditions: {v.condition_names}"
        entry += f"\n{v.document}"
        parts.append(entry)

    # GeneReviews chunks - group by nbk_id, show all chunks under the article citation
    seen_nbk: set[str] = set()
    for chunk in genereviews_chunks or []:
        c = _find_citation(citations, CitationSource.GENEREVIEWS, chunk.nbk_id)
        if not c:
            continue
        section_label = f" > {chunk.section_path}" if chunk.section_path else ""
        if chunk.nbk_id not in seen_nbk:
            entry = f"[{c.number}] GeneReviews: {chunk.condition_name} ({chunk.nbk_id}){section_label}\n{chunk.chunk_text}"
            seen_nbk.add(chunk.nbk_id)
        else:
            entry = f"[{c.number}] (continued){section_label}\n{chunk.chunk_text}"
        parts.append(entry)

    # NephroGenetics
    nephro_idx = 0
    for r in nephro_results or []:
        # Find by matching gene in title
        c = _find_citation_by_index(citations, CitationSource.NEPHROGENETICS, nephro_idx)
        nephro_idx += 1
        if not c:
            continue
        entry = f"[{c.number}] NephroGenetics: {r['gene']} - {r['title']}"
        entry += f"\nInheritance: {r['inheritance']}"
        entry += f"\nKidney manifestations: {r['kidney']}"
        if r.get("extrarenal"):
            extrarenal = r["extrarenal"]
            if isinstance(extrarenal, dict):
                extrarenal = json.dumps(extrarenal)
            entry += f"\nExtrarenal: {extrarenal}"
        if r.get("omim"):
            entry += f"\nOMIM: {r['omim']}"
        parts.append(entry)

    # gnomAD
    gnomad_idx = 0
    for g in gnomad_data or []:
        c = _find_citation_by_index(citations, CitationSource.GNOMAD, gnomad_idx)
        gnomad_idx += 1
        if not c:
            continue
        entry = f"[{c.number}] gnomAD: {g['variant_id']}"
        entry += f"\nAllele frequency: {g.get('allele_frequency', 'N/A')}"
        pops = g.get("populations")
        if pops:
            pop_parts = [f"{k}: {v}" for k, v in pops.items()]
            entry += f"\nPopulations: {', '.join(pop_parts)}"
        parts.append(entry)

    # PubMed
    pubmed_idx = 0
    for p in pubmed_abstracts or []:
        c = _find_citation_by_index(citations, CitationSource.PUBMED, pubmed_idx)
        pubmed_idx += 1
        if not c:
            continue
        abstract = p.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        entry = f"[{c.number}] PMID:{p['pmid']} - {p['title']}"
        if abstract:
            entry += f"\n{abstract}"
        parts.append(entry)

    return "\n\n".join(parts)


def postprocess_citations(response: str, citations: list[Citation]) -> str:
    """
    Post-process LLM response: replace [N] with clickable markdown links
    and append a References section.

    The LLM outputs plain [1], [2] markers. This function:
    1. Replaces [N] with [[N]](url) for clickable links in markdown
    2. Appends a formatted References section at the end
    """
    if not citations:
        return response

    by_number = {c.number: c for c in citations}

    # Strip any References section the LLM wrote itself first, so we don't end
    # up appending a second one (duplicate-References bug) and so the no-evidence
    # check below looks at the answer prose, not the model's own ref list. We
    # rebuild a canonical block (clickable links, correct titles) from our
    # citations afterwards.
    response = _LLM_REFERENCES_BLOCK.sub("", response).rstrip()

    # When the model reports that the queried variant/condition is absent from
    # the retrieved evidence, any citations attached to that answer are
    # off-target neighbours (e.g. a pronoun-only follow-up that drifted into
    # an unrelated gene). Suppressing the References block here stops those
    # stray sources from being presented as if they supported the answer. We
    # return the response with the LLM's own (stray) ref block already stripped.
    if _NO_EVIDENCE_PATTERN.search(response):
        return response

    # Find which citation numbers the LLM actually used
    used_numbers = set()
    for match in re.finditer(r'\[(\d+)\]', response):
        num = int(match.group(1))
        if num in by_number:
            used_numbers.add(num)

    if not used_numbers:
        return response

    # Replace [N] with [[N]](url) - process from highest to lowest to avoid
    # replacing [1] inside [10]
    processed = response
    for num in sorted(used_numbers, reverse=True):
        c = by_number[num]
        if c.url:
            # First: strip any LLM-generated links like [N](some_url)
            processed = re.sub(
                rf'(?<!\[)\[{num}\]\([^)]*\)',
                f'[[{num}]]({c.url})',
                processed,
            )
            # Then: replace bare [N] references
            processed = re.sub(
                rf'(?<!\[)\[{num}\](?!\()',
                f'[[{num}]]({c.url})',
                processed,
            )

    # Append References section with used citations only
    ref_lines = []
    for num in sorted(used_numbers):
        c = by_number[num]
        if c.url:
            ref_lines.append(f"- [{num}] [{c.title}]({c.url})")
        else:
            ref_lines.append(f"- [{num}] {c.title}")

    if ref_lines:
        processed += "\n\n---\n**References:**\n" + "\n".join(ref_lines)

    return processed


def _find_citation(
    citations: list[Citation], source: CitationSource, identifier: str
) -> Citation | None:
    """Find a citation by source type and identifier in title."""
    for c in citations:
        if c.source == source and identifier in c.title:
            return c
    return None


def _find_citation_by_index(
    citations: list[Citation], source: CitationSource, index: int
) -> Citation | None:
    """Find the Nth citation of a given source type."""
    count = 0
    for c in citations:
        if c.source == source:
            if count == index:
                return c
            count += 1
    return None
