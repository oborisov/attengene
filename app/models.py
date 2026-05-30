"""
Pydantic models for AttenGene.

No logic, no database code, no FastAPI imports.
"""

from enum import Enum

from pydantic import BaseModel, Field


class CitationSource(str, Enum):
    """Source database for a citation."""

    CLINVAR = "clinvar"
    GENEREVIEWS = "genereviews"
    NEPHROGENETICS = "nephrogenetics"
    GNOMAD = "gnomad"
    PUBMED = "pubmed"


class Citation(BaseModel):
    """A numbered citation linking a claim to its source."""

    number: int = Field(..., description="Citation number, e.g. [1]")
    source: CitationSource
    title: str = Field(..., description="Short display title")
    url: str = Field(..., description="Clickable link to source")
    detail: str = Field(default="", description="Extra context for LLM (not shown in references)")


class VariantEvidence(BaseModel):
    """Single variant retrieved from ClinVar."""

    variation_id: int = Field(..., description="ClinVar Variation ID")
    gene_symbol: str = Field(..., description="Gene symbol (e.g., BRCA1)")
    variant_name: str = Field(..., description="Variant name/HGVS expression")
    clinical_significance: str = Field(..., description="Pathogenicity classification")
    review_status: str = Field(..., description="ClinVar review status")
    condition_names: str | None = Field(default=None, description="Associated conditions")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    document: str = Field(..., description="Full RAG document text")
