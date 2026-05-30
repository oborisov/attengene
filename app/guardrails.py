"""
Pre- and post-generation guardrails for AttenGene.

Pre-generation:
- Validates query is not empty
- Rejects diagnosis or treatment intent (keyword-based)

Post-generation:
- Validates response contains at least one citation from evidence
- Rejects diagnosis or treatment language in response
"""

import re
from dataclasses import dataclass

from app.models import Citation, CitationSource, VariantEvidence

# Phrases indicating patient-specific diagnosis intent.
# Educational questions like "how is X diagnosed clinically?" must pass -
# they're a core GeneReviews use case, so we don't match bare "diagnose" /
# "diagnosis". Triggers below target queries about a specific person.
DIAGNOSIS_KEYWORDS = [
    "do i have",
    "does my patient have",
    "diagnose me",
    "diagnose my patient",
    "am i affected",
    "is my patient affected",
    "what disease do i have",
    "what condition do i have",
    "confirm disease",
    "confirm condition",
    "rule out",
]

# Keywords indicating treatment intent
TREATMENT_KEYWORDS = [
    "how to treat",
    "treatment for",
    "what medication",
    "which drug",
    "prescribe",
    "prescription",
    "therapy for",
    "cure for",
    "should i take",
    "dosage",
    "dose of",
    "treatment options",
    "clinical management",
    "how to manage",
]

# Phrases indicating diagnosis/treatment in LLM response
# These patterns target direct diagnostic statements to patients, not general medical descriptions
RESPONSE_DIAGNOSIS_PATTERNS = [
    r"\byou have\b(?! access| the option| provided)",  # "you have X disease" but not "you have access"
    r"\byour patient has\b",
    r"\bthis confirms\b.*\bdiagnosis\b",
    r"\bi diagnose\b",
    r"\bthe diagnosis is\b",
    r"\bbased on (?:this|these) (?:variant|result)s?,? (?:you|the patient) (?:has|have)\b",
]

RESPONSE_TREATMENT_PATTERNS = [
    r"you should take\b",
    r"i recommend\b.*\btreatment\b",
    r"prescribe\b",
    r"start\b.*\btherapy\b",
    r"the treatment is\b",
    r"take\b.*\bmedication\b",
]


@dataclass
class ValidationResult:
    """Result of guardrail validation."""

    valid: bool
    reason: str
    filtered_response: str


def validate_query(query: str) -> ValidationResult:
    """
    Pre-generation validation of user query.

    Args:
        query: User's query text

    Returns:
        ValidationResult indicating if query is acceptable
    """
    # Check empty query
    if not query or not query.strip():
        return ValidationResult(
            valid=False,
            reason="Query cannot be empty.",
            filtered_response="",
        )

    query_lower = query.lower()

    # Check diagnosis intent
    for keyword in DIAGNOSIS_KEYWORDS:
        if keyword in query_lower:
            return ValidationResult(
                valid=False,
                reason="This system cannot provide diagnoses. Please consult a qualified healthcare professional.",
                filtered_response="",
            )

    # Check treatment intent
    for keyword in TREATMENT_KEYWORDS:
        if keyword in query_lower:
            return ValidationResult(
                valid=False,
                reason="This system cannot provide treatment recommendations. Please consult a qualified healthcare professional.",
                filtered_response="",
            )

    return ValidationResult(
        valid=True,
        reason="",
        filtered_response="",
    )


def validate_response(
    response: str,
    evidence: list[VariantEvidence],
    citations: list[Citation] | None = None,
) -> ValidationResult:
    """
    Post-generation validation of LLM response.

    Args:
        response: LLM-generated response text
        evidence: Retrieved variant evidence used for generation
        citations: Numbered citations (if available, checks [N] format)

    Returns:
        ValidationResult indicating if response is acceptable
    """
    # Early return only if no evidence AND no citations
    if not evidence and not citations:
        return ValidationResult(
            valid=True,
            reason="No evidence available; response limited to insufficiency notice.",
            filtered_response=response,
        )

    response_lower = response.lower()

    # Check for diagnosis language FIRST (primary safety violation)
    for pattern in RESPONSE_DIAGNOSIS_PATTERNS:
        if re.search(pattern, response_lower):
            return ValidationResult(
                valid=False,
                reason="Response contains diagnostic language which is not permitted.",
                filtered_response=_filter_diagnostic_content(response),
            )

    # Check for treatment language SECOND
    for pattern in RESPONSE_TREATMENT_PATTERNS:
        if re.search(pattern, response_lower):
            return ValidationResult(
                valid=False,
                reason="Response contains treatment recommendations which are not permitted.",
                filtered_response=_filter_treatment_content(response),
            )

    # Check for citations LAST
    if citations:
        # New [N] citation format: check that at least one [N] appears
        # where N maps to a known citation number
        citation_numbers = {c.number for c in citations}
        found_bracket = any(
            f"[{n}]" in response for n in citation_numbers
        )
        if not found_bracket:
            return ValidationResult(
                valid=False,
                reason="Response must cite at least one source using [N] format.",
                filtered_response=_add_citation_reminder(
                    response, evidence, citations=citations,
                ),
            )
    else:
        # Legacy: accept any mention of the variation ID (flexible format)
        evidence_ids = {str(v.variation_id) for v in evidence}
        found_citation = any(vid in response for vid in evidence_ids)

        if not found_citation:
            return ValidationResult(
                valid=False,
                reason="Response must cite at least one ClinVar variation ID from the provided evidence.",
                filtered_response=_add_citation_reminder(response, evidence),
            )

    return ValidationResult(
        valid=True,
        reason="",
        filtered_response=response,
    )


def _add_citation_reminder(
    response: str,
    evidence: list[VariantEvidence],
    citations: list[Citation] | None = None,
) -> str:
    """Replace uncited response with a grounded evidence summary."""
    # Build a simple evidence listing as fallback
    if citations:
        lines = ["FINDINGS:"]
        for c in citations:
            lines.append(f"- [{c.number}] {c.title}")
        return "\n".join(lines)
    # Legacy format
    lines = ["The following ClinVar evidence was retrieved:"]
    for v in evidence[:5]:
        lines.append(f"- ClinVar:{v.variation_id} - {v.gene_symbol} {v.variant_name}, {v.clinical_significance}")
    return "\n".join(lines)


def _filter_diagnostic_content(response: str) -> str:
    """Filter diagnostic language from response."""
    return (
        "I cannot provide diagnostic information. "
        "The retrieved ClinVar evidence is for research purposes only. "
        "Please consult a qualified healthcare professional for diagnosis."
    )


def _filter_treatment_content(response: str) -> str:
    """Filter treatment language from response."""
    return (
        "I cannot provide treatment recommendations. "
        "The retrieved ClinVar evidence is for research purposes only. "
        "Please consult a qualified healthcare professional for treatment guidance."
    )


if __name__ == "__main__":
    # Test pre-generation
    print("=== Pre-generation tests ===")

    test_queries = [
        "What pathogenic variants exist in BRCA1?",
        "Do I have Lynch syndrome?",
        "What treatment should I take for hemochromatosis?",
        "",
    ]

    for q in test_queries:
        result = validate_query(q)
        status = "PASS" if result.valid else "REJECT"
        print(f"[{status}] '{q[:50]}...' - {result.reason if result.reason else 'OK'}")

    # Test post-generation
    print("\n=== Post-generation tests ===")

    mock_evidence = [
        VariantEvidence(
            variation_id=12345,
            gene_symbol="BRCA1",
            variant_name="c.123A>G",
            clinical_significance="Pathogenic",
            review_status="reviewed by expert panel",
            condition_names="Breast cancer",
            similarity=0.85,
            document="Test document",
        )
    ]

    test_responses = [
        "The variant ClinVar:12345 in BRCA1 is classified as pathogenic.",
        "This variant is pathogenic but I forgot to cite the ID.",
        "You have breast cancer based on this variant.",
        "I recommend you start chemotherapy treatment.",
    ]

    for r in test_responses:
        result = validate_response(r, mock_evidence)
        status = "PASS" if result.valid else "REJECT"
        print(f"[{status}] '{r[:50]}...' - {result.reason if result.reason else 'OK'}")
