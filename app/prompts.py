"""
Prompt templates for AttenGene RAG.

System prompt enforces:
- ClinVar-only evidence
- Mandatory citation of ClinVar variation IDs
- Explicit uncertainty when evidence is insufficient
- No diagnosis
- No treatment recommendations

User prompt injects retrieved evidence with provenance.
GeneReviews links are added for clinical context.
"""

from app.models import VariantEvidence
from app.genereviews import format_genereviews_context

SYSTEM_PROMPT = """You are a clinical genetics research assistant. Your role is to help users explore variant evidence from ClinVar and provide links to GeneReviews for clinical context.

STRICT RULES:
1. Only use information from the provided ClinVar evidence. Do not use external knowledge about variants.
2. Every factual claim about a variant MUST include at least one ClinVar variation ID. When citing variants, always use the exact format ClinVar:<variation_id> (e.g., ClinVar:12345).
3. If the evidence is insufficient to answer the question, explicitly state: "The provided evidence does not contain sufficient information to answer this question."
4. Never provide diagnoses or suggest that a patient has a specific condition.
5. Never recommend treatments, medications, or clinical actions.
6. Never interpret variants beyond what is stated in ClinVar.
7. If asked for medical advice, respond: "I cannot provide medical advice. Please consult a qualified healthcare professional."
8. Do not mention variants that are not explicitly listed in the provided ClinVar evidence.
9. When GeneReviews links are provided, include them in your response for clinical management guidelines.

Your responses should:
- Summarize relevant variant evidence from ClinVar
- Cite specific ClinVar variation IDs
- Include GeneReviews links when available for clinical context
- Acknowledge limitations and uncertainty
- Be factual and evidence-based only"""


def build_user_prompt(query: str, evidence: list[VariantEvidence]) -> str:
    """
    Build user prompt with retrieved evidence and GeneReviews context.

    Args:
        query: User's original query
        evidence: Retrieved variant evidence from ClinVar

    Returns:
        Formatted user prompt with evidence context
    """
    # Extract unique gene symbols from evidence
    gene_symbols = list(set(v.gene_symbol for v in evidence if v.gene_symbol))

    # Get GeneReviews context
    genereviews_context = format_genereviews_context(gene_symbols, query)

    if not evidence:
        return f"""User question: {query}

No relevant ClinVar evidence was retrieved for this query.
{genereviews_context}

Please inform the user that no matching variants were found in ClinVar.
If GeneReviews links are provided above, include them for general information about the gene/condition."""

    evidence_text = _format_evidence(evidence)

    return f"""User question: {query}

Retrieved ClinVar evidence:
{evidence_text}
{genereviews_context}

Based on the above ClinVar evidence and clinical context, answer the user's question.
- Cite ClinVar variation IDs for any variants you mention (format: ClinVar:12345)
- Include GeneReviews links if provided above for clinical management information
- Do not invent information not present in the evidence"""


def _format_evidence(evidence: list[VariantEvidence]) -> str:
    """Format evidence list for prompt injection."""
    parts = []

    for i, v in enumerate(evidence, 1):
        entry = f"""--- Evidence {i} ---
ClinVar ID: {v.variation_id}
Gene: {v.gene_symbol}
Variant: {v.variant_name}
Clinical Significance: {v.clinical_significance}
Review Status: {v.review_status}"""

        if v.condition_names:
            entry += f"\nConditions: {v.condition_names}"

        entry += f"\nDetails: {v.document}"
        parts.append(entry)

    return "\n\n".join(parts)


def build_messages(query: str, evidence: list[VariantEvidence]) -> list[dict[str, str]]:
    """
    Build complete message list for LLM.

    Args:
        query: User's original query
        evidence: Retrieved variant evidence

    Returns:
        OpenAI-format messages list
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, evidence)},
    ]


# --- Unified prompt building for OpenAI-compatible endpoint ---

UNIFIED_SYSTEM_PROMPT = """You extract findings from retrieved evidence. You have NO medical knowledge of your own.

OUTPUT FORMAT (mandatory):
FINDINGS:
- [N] one finding from evidence
- [N] one finding from evidence
GAPS: what the query asked that evidence does not cover

RULES:
- Every line in FINDINGS must start with a citation number like [1], [2], [3]
- Only state what the evidence literally says. Do NOT add explanations, guidelines, percentages, or recommendations from your own knowledge.
- Do NOT create tables, headers, or sections. Only bullet points.
- If you are unsure whether a fact is in the evidence or from your knowledge, leave it out.
- Maximum 15 bullet points.

Example:
Query: BRCA1 pathogenic variants
FINDINGS:
- [1] BRCA1 c.5266dupC is classified as pathogenic, reviewed by expert panel
- [2] BRCA1 c.68_69delAG is classified as pathogenic, associated with hereditary breast and ovarian cancer
GAPS: No population frequency data in retrieved evidence"""

RETRY_SYSTEM_PROMPT = """Your previous response was rejected. You MUST use this format:
FINDINGS:
- [1] fact from evidence
- [2] fact from evidence
GAPS: what is missing

No tables. No paragraphs. No headers. Only the format above."""


def build_augmented_messages(
    conversation: list[dict[str, str]],
    retrieved_context: str,
) -> list[dict[str, str]]:
    """
    Build message list for the LLM with RAG context injected.

    Instructions come first to set the format constraint firmly,
    then evidence is placed at the end closest to generation.

    Args:
        conversation: Full conversation from client (system + user + assistant messages)
        retrieved_context: Combined context from route_and_retrieve()

    Returns:
        Messages list ready for the LLM
    """
    if retrieved_context:
        system_content = f"{UNIFIED_SYSTEM_PROMPT}\n\nRETRIEVED EVIDENCE:\n{retrieved_context}"
    else:
        system_content = f"{UNIFIED_SYSTEM_PROMPT}\n\nNo relevant evidence was retrieved for this query."

    # Build messages: RAG system prompt + conversation history (skip client's system message)
    messages = [{"role": "system", "content": system_content}]
    for msg in conversation:
        if msg["role"] != "system":
            messages.append({"role": msg["role"], "content": msg["content"]})

    return messages
