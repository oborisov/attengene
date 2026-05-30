"""
GeneReviews lookup module.

Maps gene symbols to GeneReviews entries and provides clinical context.
Data source: ftp://ftp.ncbi.nih.gov/pub/GeneReviews/

Copyright Notice: GeneReviews content is University of Washington 1993-2025.
Used under permission for noncommercial research purposes.
See: https://www.ncbi.nlm.nih.gov/books/NBK1116/ for full terms.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeneReviewsEntry:
    """A GeneReviews entry linking a gene to a disease."""

    shortname: str
    nbk_id: str
    gene: str
    disease: str

    @property
    def url(self) -> str:
        """NCBI GeneReviews URL."""
        return f"https://www.ncbi.nlm.nih.gov/books/{self.nbk_id}/"


class GeneReviewsDB:
    """GeneReviews database for gene-disease lookups."""

    def __init__(self, data_path: str | None = None):
        """
        Initialize GeneReviews database.

        Args:
            data_path: Path to GRshortname_NBKid_genesymbol_dzname.txt
                      If None, uses default location in data/genereviews/
        """
        if data_path is None:
            # Try multiple locations
            candidates = [
                Path(__file__).parent.parent / "data" / "genereviews" / "GRshortname_NBKid_genesymbol_dzname.txt",
                Path("/app/data/genereviews/GRshortname_NBKid_genesymbol_dzname.txt"),  # Docker
                Path("data/genereviews/GRshortname_NBKid_genesymbol_dzname.txt"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    data_path = str(candidate)
                    break

        self.entries: list[GeneReviewsEntry] = []
        self._gene_index: dict[str, list[GeneReviewsEntry]] = {}
        self._disease_index: dict[str, list[GeneReviewsEntry]] = {}

        if data_path and os.path.exists(data_path):
            self._load(data_path)

    def _load(self, path: str) -> None:
        """Load GeneReviews data from pipe-delimited file."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) != 4:
                    continue

                shortname, nbk_id, gene, disease = parts

                # Skip "Not applicable" genes
                if gene == "Not applicable":
                    continue

                entry = GeneReviewsEntry(
                    shortname=shortname,
                    nbk_id=nbk_id,
                    gene=gene.upper(),
                    disease=disease,
                )
                self.entries.append(entry)

                # Index by gene
                gene_upper = gene.upper()
                if gene_upper not in self._gene_index:
                    self._gene_index[gene_upper] = []
                self._gene_index[gene_upper].append(entry)

                # Index by disease keywords (lowercase for case-insensitive search)
                disease_lower = disease.lower()
                if disease_lower not in self._disease_index:
                    self._disease_index[disease_lower] = []
                self._disease_index[disease_lower].append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def lookup_by_gene(self, gene_symbol: str) -> list[GeneReviewsEntry]:
        """
        Find GeneReviews entries for a gene symbol.

        Args:
            gene_symbol: Gene symbol (e.g., "BRCA1", "FBN1")

        Returns:
            List of matching GeneReviews entries
        """
        return self._gene_index.get(gene_symbol.upper(), [])

    def lookup_by_disease(self, search_term: str) -> list[GeneReviewsEntry]:
        """
        Find GeneReviews entries matching a disease/syndrome name.

        Args:
            search_term: Disease or syndrome name (partial match, case-insensitive)

        Returns:
            List of matching GeneReviews entries
        """
        search_lower = search_term.lower()
        results = []
        seen_nbk_ids = set()

        for disease, entries in self._disease_index.items():
            if search_lower in disease:
                for entry in entries:
                    if entry.nbk_id not in seen_nbk_ids:
                        results.append(entry)
                        seen_nbk_ids.add(entry.nbk_id)

        return results

    def format_gene_links(self, gene_symbol: str) -> str | None:
        """
        Format GeneReviews links for a gene as markdown.

        Args:
            gene_symbol: Gene symbol

        Returns:
            Markdown-formatted links or None if no entries found
        """
        entries = self.lookup_by_gene(gene_symbol)
        if not entries:
            return None

        lines = [f"**{gene_symbol} GeneReviews:**"]
        for entry in entries:
            lines.append(f"- {entry.disease}: {entry.url}")

        return "\n".join(lines)

    def format_disease_lookup(self, search_term: str) -> str | None:
        """
        Format disease lookup results as markdown.

        Args:
            search_term: Disease or syndrome name

        Returns:
            Markdown-formatted results or None if no entries found
        """
        entries = self.lookup_by_disease(search_term)
        if not entries:
            return None

        # Group by NBK ID (disease)
        by_disease: dict[str, list[GeneReviewsEntry]] = {}
        for entry in entries:
            if entry.nbk_id not in by_disease:
                by_disease[entry.nbk_id] = []
            by_disease[entry.nbk_id].append(entry)

        lines = []
        for nbk_id, disease_entries in by_disease.items():
            disease = disease_entries[0].disease
            genes = sorted(set(e.gene for e in disease_entries))
            url = disease_entries[0].url

            lines.append(f"**{disease}**")
            lines.append(f"- Genes: {', '.join(genes)}")
            lines.append(f"- GeneReviews: {url}")
            lines.append("")

        return "\n".join(lines).strip() if lines else None


# Global instance (lazy loaded)
_db: GeneReviewsDB | None = None


def get_genereviews_db() -> GeneReviewsDB:
    """Get the global GeneReviews database instance."""
    global _db
    if _db is None:
        _db = GeneReviewsDB()
    return _db



def format_genereviews_context(gene_symbols: list[str], query: str = "") -> str:
    """
    Build GeneReviews context for RAG prompt.

    Args:
        gene_symbols: List of gene symbols detected in query/results
        query: Original user query (for disease lookup)

    Returns:
        Formatted GeneReviews context string
    """
    db = get_genereviews_db()
    if len(db) == 0:
        return ""

    sections = []

    # Lookup by disease/syndrome in query
    disease_keywords = [
        "syndrome", "disease", "disorder", "cancer", "cardiomyopathy",
        "dystrophy", "anemia", "hemophilia", "thalassemia",
    ]
    if any(kw in query.lower() for kw in disease_keywords):
        disease_info = db.format_disease_lookup(query)
        if disease_info:
            sections.append(f"**Matching syndromes/diseases:**\n\n{disease_info}")

    # Lookup by gene symbols (limit to first 3 to avoid context overflow)
    for gene in gene_symbols[:3]:
        gene_info = db.format_gene_links(gene)
        if gene_info:
            sections.append(gene_info)

    if not sections:
        return ""

    return (
        "\n\n**GeneReviews Clinical Information ( University of Washington):**\n\n"
        + "\n\n---\n\n".join(sections)
    )
