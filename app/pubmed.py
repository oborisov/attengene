"""
PubMed E-utilities live search.

Searches PubMed for relevant abstracts given a gene symbol and optional
condition. Uses the NCBI E-utilities API (esearch + efetch).

Rate limit: 3 requests/second without API key. Set NCBI_API_KEY env var
for 10 req/sec.
"""

import logging
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TIMEOUT = 10.0
_MAX_ABSTRACT_LEN = 300

_api_key = os.environ.get("NCBI_API_KEY", "")


def search_pubmed(
    gene: str,
    condition: str = "",
    max_results: int = 3,
) -> list[dict]:
    """
    Search PubMed for relevant abstracts.

    Args:
        gene: Gene symbol (e.g. "BRCA1")
        condition: Optional condition/phenotype to narrow results
        max_results: Maximum number of results to return

    Returns:
        List of dicts with keys: pmid, title, abstract, url
    """
    # Build search query
    terms = [f"{gene}[Gene]", "pathogenic variant"]
    if condition:
        terms.append(condition)
    query = " AND ".join(terms)

    try:
        pmids = _esearch(query, max_results)
        if not pmids:
            return []
        return _efetch(pmids)
    except Exception:
        logger.warning("PubMed search failed for gene=%s", gene, exc_info=True)
        return []


def _esearch(query: str, max_results: int) -> list[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "sort": "relevance",
        "retmode": "json",
    }
    if _api_key:
        params["api_key"] = _api_key

    url = f"{_BASE}/esearch.fcgi"
    resp = httpx.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    result = data.get("esearchresult", {})
    return result.get("idlist", [])


def _efetch(pmids: list[str]) -> list[dict]:
    """Fetch article details (title, abstract) for given PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if _api_key:
        params["api_key"] = _api_key

    url = f"{_BASE}/efetch.fcgi"
    resp = httpx.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    results = []
    root = ET.fromstring(resp.text)

    for article_el in root.findall(".//PubmedArticle"):
        pmid_el = article_el.find(".//PMID")
        title_el = article_el.find(".//ArticleTitle")
        abstract_el = article_el.find(".//AbstractText")

        if pmid_el is None or title_el is None:
            continue

        pmid = pmid_el.text or ""
        title = title_el.text or ""
        abstract = abstract_el.text or "" if abstract_el is not None else ""

        # Truncate abstract
        if len(abstract) > _MAX_ABSTRACT_LEN:
            abstract = abstract[:_MAX_ABSTRACT_LEN] + "..."

        results.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return results
