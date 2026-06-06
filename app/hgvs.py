"""
HGVS variant-string parsing for ClinVar exact lookup.

The chat router originally fed the entire user message into a pg_trgm fuzzy
search against ClinVar variant names. That dilutes the signal: a conversational
query like "what about the variant c.526G>A, p.(Ala176Thr) in ALPL" trigram-
matches the target name at only ~0.38 (vs ~0.61 for the bare HGVS token),
because most of the sentence is filler. A fixed similarity floor then can't
separate a real match from coincidental noise.

This module extracts the structured pieces - gene symbol and HGVS tokens - so
retrieval can do an *exact* substring lookup on the parsed token instead of a
fuzzy match on the sentence. Exact lookup gives high precision and clean true
negatives (variant genuinely absent -> no rows -> "not in evidence"), with no
cross-gene bleed.

Coverage is intentionally pragmatic, seeded from real test queries; extend the
patterns as clinical colleagues supply more input formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Coding / genomic / mito / non-coding HGVS at the nucleotide level, e.g.
#   c.526G>A   c.68_69delAG   c.5266dupC   c.1559delT   g.123A>T   m.1555A>G
# We capture the leading "<type>.<rest>" up to the next comma/paren/space that
# is not part of the token. ">" may arrive HTML-escaped as "&gt;".
_NUCLEOTIDE_HGVS = re.compile(
    r"(?<![A-Za-z0-9])"          # not mid-identifier
    r"([cgmn]\.\s*"              # type prefix
    r"[0-9][0-9_+\-*]*"          # position(s): 526, 68_69, 5266, 1223+1
    r"(?:[ACGT]+)?"              # optional ref base(s)
    r"(?:(?:>|&gt;)[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+|delins[ACGT]+)?"
    r")",
    re.IGNORECASE,
)

# Protein-level HGVS, e.g. p.Ala176Thr  p.(Ala176Thr)  p.Leu520ArgfsTer86
# Accepts 3-letter or 1-letter amino acids, optional parentheses, fs/Ter.
_PROTEIN_HGVS = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(p\.\(?"
    r"[A-Z][a-z]{0,2}"          # ref AA (Ala / A)
    r"[0-9]+"                    # position
    r"(?:[A-Z][a-z]{0,2}|=|\*|Ter|fs(?:Ter)?[0-9]*|del|dup)?"  # alt AA / event
    r"\)?)",
    re.IGNORECASE,
)

# Transcript / reference accession prefix, e.g. NM_000478.6  NC_000001.11
_TRANSCRIPT = re.compile(r"\b([NX][MRCG]_[0-9]+\.[0-9]+)\b")

# dbSNP rsID
_RSID = re.compile(r"\b(rs[0-9]+)\b", re.IGNORECASE)

# Gene symbol (mirrors router._GENE_PATTERN / _NOT_GENES). Imported lazily to
# avoid a circular import with router.
_GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")


@dataclass
class ParsedVariant:
    """Structured pieces extracted from a free-text variant query."""

    gene: str | None = None
    c_hgvs: str | None = None       # normalized coding HGVS, e.g. "c.526G>A"
    p_hgvs: str | None = None       # normalized protein HGVS, e.g. "p.Ala176Thr"
    transcript: str | None = None
    rsid: str | None = None
    raw_tokens: list[str] = field(default_factory=list)

    @property
    def has_variant_token(self) -> bool:
        """True if any concrete variant identifier was found (not just a gene)."""
        return any((self.c_hgvs, self.p_hgvs, self.rsid))


def _normalize_hgvs(token: str) -> str:
    """Canonicalize an HGVS token for substring matching against ClinVar names.

    ClinVar stores names like 'NM_000478.6(ALPL):c.526G>A (p.Ala176Thr)', so we
    normalize to that surface form: unescape &gt;, uppercase the type prefix and
    bases, strip internal whitespace, drop wrapping parens on protein tokens.
    """
    t = token.strip()
    t = t.replace("&gt;", ">").replace("&GT;", ">")
    t = re.sub(r"\s+", "", t)
    # Lowercase prefix (c./p./g.) is fine, but bases and '>' must be uppercase
    # to match ClinVar; do a targeted uppercase of base letters around '>'/del/dup.
    # Simplest robust approach: uppercase the whole token except we keep the
    # 3-letter amino acid casing for protein tokens (Ala, not ALA).
    if t[:2].lower() == "p.":
        t = t.strip("()")
        t = "p." + t[2:].lstrip("(").rstrip(")")
        return t
    # Nucleotide token: uppercase prefix+bases to match ClinVar, but the event
    # keywords del/dup/ins/delins are stored LOWERCASE in ClinVar names
    # (e.g. c.1559delT, c.5266dupC), so case-fold them back down.
    t = t.upper()
    t = t[0].lower() + t[1:]          # type prefix: C. -> c.
    for ev in ("DELINS", "DEL", "DUP", "INS"):
        t = t.replace(ev, ev.lower())
    return t


def parse_variant(query: str, not_genes: set[str] | None = None) -> ParsedVariant:
    """Extract gene symbol + HGVS tokens from a free-text query.

    Returns a ParsedVariant. `has_variant_token` is False when nothing more than
    a gene (or nothing at all) was found, in which case callers should fall back
    to fuzzy/hybrid retrieval rather than an exact lookup.
    """
    pv = ParsedVariant()

    if not_genes is None:
        from app.router import _NOT_GENES as not_genes  # lazy, avoid cycle

    c_match = _NUCLEOTIDE_HGVS.search(query)
    if c_match:
        tok = c_match.group(1)
        pv.raw_tokens.append(tok)
        norm = _normalize_hgvs(tok)
        # route c./g./m./n. - we only treat c. as coding for the c_hgvs slot;
        # keep others in raw_tokens for the exact lookup to still use.
        if norm[:2].lower() == "c.":
            pv.c_hgvs = norm

    p_match = _PROTEIN_HGVS.search(query)
    if p_match:
        tok = p_match.group(1)
        pv.raw_tokens.append(tok)
        pv.p_hgvs = _normalize_hgvs(tok)

    tx = _TRANSCRIPT.search(query)
    if tx:
        pv.transcript = tx.group(1)

    rs = _RSID.search(query)
    if rs:
        pv.rsid = rs.group(1).lower()

    # Gene symbol: first all-caps token that isn't a stopword and isn't itself
    # part of an HGVS/transcript token already captured.
    for cand in _GENE_PATTERN.findall(query):
        if cand in not_genes or len(cand) < 2:
            continue
        pv.gene = cand
        break

    return pv
