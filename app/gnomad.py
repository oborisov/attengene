"""
gnomAD GraphQL live lookup.

Queries the gnomAD API for population allele frequencies given a variant
in gnomAD format (chrom-pos-ref-alt). Only attempts variants where the
ClinVar name field contains chromosomal notation (NC_* references) -
complex HGVS is skipped.

API docs: https://gnomad.broadinstitute.org/api
"""

import logging
import re

import httpx

logger = logging.getLogger(__name__)

_GNOMAD_API = "https://gnomad.broadinstitute.org/api"
_TIMEOUT = 10.0

# Regex for chromosomal HGVS notation like NC_000017.11:g.43057051T>C
_CHROMOSOMAL_RE = re.compile(
    r"NC_0000(\d{2})\.\d+:g\.(\d+)([ACGT]+)>([ACGT]+)"
)

# NC accession chromosome number mapping (leading zeros stripped)
# NC_000001 -> 1, NC_000023 -> X, NC_000024 -> Y
_CHROM_MAP = {
    "23": "X",
    "24": "Y",
}

_QUERY = """
query GnomadVariant($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
    variant_id
    genome {
      ac
      an
      af
      populations {
        id
        ac
        an
        af
      }
    }
    exome {
      ac
      an
      af
      populations {
        id
        ac
        an
        af
      }
    }
  }
}
"""


def parse_gnomad_id(variant_name: str) -> str | None:
    """
    Parse a ClinVar variant name to gnomAD format (chrom-pos-ref-alt).

    Only handles simple SNVs with chromosomal notation (NC_* references).
    Returns None for complex HGVS or unparseable names.

    Examples:
        "NC_000017.11:g.43057051T>C" -> "17-43057051-T-C"
        "NM_007294.4(BRCA1):c.5266dupC" -> None (transcript notation)
    """
    match = _CHROMOSOMAL_RE.search(variant_name)
    if not match:
        return None

    chrom_num = match.group(1).lstrip("0") or "0"
    chrom = _CHROM_MAP.get(chrom_num, chrom_num)
    pos = match.group(2)
    ref = match.group(3)
    alt = match.group(4)

    return f"{chrom}-{pos}-{ref}-{alt}"


def fetch_variant_frequency(variant_id: str) -> dict | None:
    """
    Query gnomAD GraphQL API for population allele frequencies.

    Args:
        variant_id: gnomAD format "chrom-pos-ref-alt" (e.g. "17-43057051-T-C")

    Returns:
        Dict with keys: variant_id, allele_frequency, populations, url
        None if variant not found or API error.
    """
    try:
        resp = httpx.post(
            _GNOMAD_API,
            json={
                "query": _QUERY,
                "variables": {
                    "variantId": variant_id,
                    "dataset": "gnomad_r4",
                },
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("gnomAD lookup failed for %s", variant_id, exc_info=True)
        return None

    variant = data.get("data", {}).get("variant")
    if not variant:
        return None

    # Prefer genome data, fall back to exome
    freq_data = variant.get("genome") or variant.get("exome")
    if not freq_data:
        return None

    af = freq_data.get("af")
    if af is None:
        return None

    # Extract population frequencies
    populations = {}
    for pop in freq_data.get("populations", []):
        pop_id = pop.get("id", "")
        pop_af = pop.get("af")
        if pop_af is not None and pop_af > 0 and pop_id:
            populations[pop_id] = round(pop_af, 6)

    return {
        "variant_id": variant_id,
        "allele_frequency": round(af, 6),
        "populations": populations,
        "url": f"https://gnomad.broadinstitute.org/variant/{variant_id}?dataset=gnomad_r4",
    }
