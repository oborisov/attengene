"""
Audit logging for AttenGene queries.

Inserts into audit_logs table for compliance (GDPR, 6-10 year retention).
"""

import logging

from psycopg2.extras import Json

from app.db import get_connection, release_connection
from app.models import VariantEvidence

logger = logging.getLogger(__name__)


def log_query(
    session_id: str,
    user_id: str,
    query: str,
    response: str,
    evidence: list[VariantEvidence],
    model_used: str,
    latency_ms: int,
    client_ip: str | None = None,
    query_type: str | None = None,
    was_rejected: bool = False,
    rejection_reason: str | None = None,
    error_message: str | None = None,
) -> None:
    """
    Log a query to the audit_logs table.

    Args:
        session_id: Session identifier
        user_id: User identifier
        client_ip: Client IP address
        query: Original query text
        response: LLM response text
        evidence: Retrieved variant evidence
        model_used: LLM model identifier
        latency_ms: Total processing time in milliseconds
        was_rejected: Whether query was rejected by guardrails
        rejection_reason: Reason for rejection (if rejected)
        error_message: Error message (if error occurred)
    """
    # Build retrieval scores JSON
    retrieval_scores = [
        {"id": v.variation_id, "score": round(v.similarity, 4)}
        for v in evidence
    ]

    # Extract variant IDs as array
    retrieved_variant_ids = [v.variation_id for v in evidence]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs (
                    session_id,
                    user_id,
                    client_ip,
                    query_text,
                    query_type,
                    retrieved_variant_ids,
                    retrieval_scores,
                    response_text,
                    was_rejected,
                    rejection_reason,
                    model_name,
                    model_version,
                    total_time_ms,
                    error_message
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    session_id,
                    user_id,
                    client_ip,
                    query,
                    query_type,
                    retrieved_variant_ids,
                    Json(retrieval_scores),
                    response,
                    was_rejected,
                    rejection_reason,
                    model_used,
                    "gpu",  # model_version: deployment type
                    latency_ms,
                    error_message,
                ),
            )
        conn.commit()
    finally:
        release_connection(conn)


def log_auth_event(
    event_type: str,
    client_ip: str | None = None,
    user_agent: str | None = None,
    endpoint: str | None = None,
    detail: str | None = None,
) -> None:
    """
    Log an authentication event. Fire-and-forget - errors are logged but
    never block the request.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO auth_events (event_type, client_ip, user_agent, endpoint, detail)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (event_type, client_ip, user_agent, endpoint, detail),
                )
            conn.commit()
        finally:
            release_connection(conn)
    except Exception:
        logger.warning("Failed to log auth event: %s", event_type, exc_info=True)


if __name__ == "__main__":
    # Test logging
    from uuid import uuid4

    test_evidence = [
        VariantEvidence(
            variation_id=12345,
            gene_symbol="BRCA1",
            variant_name="c.123A>G",
            clinical_significance="Pathogenic",
            review_status="reviewed by expert panel",
            condition_names="Breast cancer",
            similarity=0.8523,
            document="Test document",
        ),
        VariantEvidence(
            variation_id=67890,
            gene_symbol="BRCA2",
            variant_name="c.456T>C",
            clinical_significance="Likely pathogenic",
            review_status="criteria provided",
            condition_names=None,
            similarity=0.7891,
            document="Test document 2",
        ),
    ]

    log_query(
        session_id=str(uuid4()),
        user_id="test_user",
        client_ip="127.0.0.1",
        query="What pathogenic variants exist in BRCA1?",
        response="Based on ClinVar evidence, BRCA1 variant ClinVar:12345 is pathogenic.",
        evidence=test_evidence,
        model_used="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        latency_ms=1234,
    )

    print("Audit log entry created successfully.")

    # Verify
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, session_id, query_text, retrieval_scores FROM audit_logs ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            print(f"ID: {row[0]}")
            print(f"Session: {row[1]}")
            print(f"Query: {row[2]}")
            print(f"Scores: {row[3]}")
    finally:
        release_connection(conn)
