"""
OpenAI-compatible API endpoints.

/v1/models           - lists active generation backends (attengene-local,
                       attengene-mistral, ...). OWUI uses this to populate
                       the chat-dropdown model picker.
/v1/chat/completions - dispatches by `request.model` to the matching
                       backend. The RAG pipeline (retrieve, guardrails,
                       prompt, citations, audit) runs identically across
                       backends; only the final HTTP call differs.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import verify_api_key
from app.audit import log_query
from app.citations import postprocess_citations, REFERENCES_HEADING
from app.guardrails import validate_query, validate_response
from app.llm import (
    UnknownBackendError,
    available_model_ids,
    generate,
    generate_stream,
)
from app.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    DeltaMessage,
    StreamChoice,
    Usage,
)
from app.prompts import build_augmented_messages, RETRY_SYSTEM_PROMPT
from app.router import route_and_retrieve

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])

# Per-model description surfaced in /v1/models. Cloud-backed entries
# carry an explicit non-clinical warning - chats sent to them leave the
# host and shift the GDPR boundary.
_MODEL_DESCRIPTIONS = {
    "attengene-local": "AttenGene RAG with local llama-server backend (GDPR-safe).",
    "attengene-mistral": "AttenGene RAG with Mistral cloud backend. Non-clinical only - chats leave the box.",
    "attengene-claude": "AttenGene RAG with Anthropic Claude backend. Non-clinical only - chats leave the box.",
}


@router.get("/models")
async def list_models():
    """Advertise every configured backend as a separate OpenAI model id."""
    now = int(time.time())
    data = []
    for model_id in available_model_ids():
        entry = {
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "attengene",
        }
        description = _MODEL_DESCRIPTIONS.get(model_id)
        if description:
            entry["description"] = description
        data.append(entry)
    return {"object": "list", "data": data}


def _unknown_model_response(model_id: str) -> JSONResponse:
    """OpenAI-shaped error for a model id that isn't in BACKENDS."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": (
                    f"Unknown model: {model_id!r}. "
                    f"Available models: {available_model_ids()}"
                ),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completion with RAG pipeline.

    Flow:
    1. Validate the requested model id (dispatch target).
    2. Extract last user message for retrieval.
    3. Pre-generation guardrails.
    4. Route and retrieve from relevant databases.
    5. Build augmented messages with context.
    6. Forward to the selected backend.
    7. Post-generation validation + audit.
    8. Return in OpenAI-compatible format.
    """
    model_id = request.model
    if model_id not in available_model_ids():
        return _unknown_model_response(model_id)

    start_time = time.perf_counter()
    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    client_ip = req.client.host if req.client else None

    # Extract last user message for retrieval
    user_query = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    if not user_query:
        return ChatCompletionResponse(
            id=completion_id,
            model=model_id,
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant",
                        content="No user message found in the conversation.",
                    ),
                )
            ],
        )

    # Skip Open WebUI metadata requests (title generation, tags, follow-ups).
    # These are not clinical queries and should not be logged or processed.
    if user_query.lstrip().startswith("### Task:"):
        empty_response = "AttenGene clinical genetics assistant"
        if request.stream:
            return _stream_text(completion_id, model_id, empty_response)
        return ChatCompletionResponse(
            id=completion_id,
            model=model_id,
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant",
                        content=empty_response,
                    ),
                )
            ],
        )

    # Step 3: Pre-generation guardrails
    query_validation = validate_query(user_query)
    if not query_validation.valid:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_query(
            session_id=completion_id,
            user_id="openai-api",
            query=user_query,
            response=query_validation.reason,
            evidence=[],
            model_used=model_id,
            latency_ms=latency_ms,
            client_ip=client_ip,
            was_rejected=True,
            rejection_reason=query_validation.reason,
        )

        if request.stream:
            return _stream_text(completion_id, model_id, query_validation.reason)

        return ChatCompletionResponse(
            id=completion_id,
            model=model_id,
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant",
                        content=query_validation.reason,
                    ),
                )
            ],
        )

    # Build conversation history for LLM
    conversation_history = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]

    # Streaming: move retrieval inside generator so we can show status
    if request.stream:
        return _stream_response(
            completion_id=completion_id,
            model_id=model_id,
            conversation_history=conversation_history,
            user_query=user_query,
            client_ip=client_ip,
            start_time=start_time,
        )

    # Non-streaming: retrieve first
    retrieval = route_and_retrieve(user_query, k=5)
    llm_messages = build_augmented_messages(
        conversation_history, retrieval.prompt_context
    )

    try:
        llm_response = await generate(llm_messages, model_id)
        response_text = llm_response.response_text
    except UnknownBackendError:
        # Backend table changed between the /v1/models check and now.
        return _unknown_model_response(model_id)
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_query(
            session_id=completion_id,
            user_id="openai-api",
            query=user_query,
            response="",
            evidence=retrieval.clinvar_evidence,
            model_used=model_id,
            latency_ms=latency_ms,
            client_ip=client_ip,
            query_type=retrieval.query_type,
            error_message=str(e),
        )
        return ChatCompletionResponse(
            id=completion_id,
            model=model_id,
            choices=[
                Choice(
                    message=ChatMessage(
                        role="assistant",
                        content=f"LLM service unavailable: {e}",
                    ),
                )
            ],
        )

    # Post-generation validation with retry
    response_validation = validate_response(
        response_text, retrieval.clinvar_evidence,
        citations=retrieval.citations,
    )
    if not response_validation.valid and retrieval.citations:
        retry_messages = [
            {"role": "system", "content": RETRY_SYSTEM_PROMPT + "\n\n" + retrieval.prompt_context},
            {"role": "user", "content": user_query},
        ]
        try:
            retry_response = await generate(retry_messages, model_id)
            retry_validation = validate_response(
                retry_response.response_text, retrieval.clinvar_evidence,
                citations=retrieval.citations,
            )
            if retry_validation.valid:
                response_text = retry_response.response_text
                response_validation = retry_validation
            else:
                response_text = response_validation.filtered_response
        except Exception:
            response_text = response_validation.filtered_response
    elif not response_validation.valid:
        response_text = response_validation.filtered_response

    # Post-process: add clickable links and References section
    response_text = postprocess_citations(response_text, retrieval.citations)

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    log_query(
        session_id=completion_id,
        user_id="openai-api",
        query=user_query,
        response=response_text,
        evidence=retrieval.clinvar_evidence,
        model_used=model_id,
        latency_ms=latency_ms,
        client_ip=client_ip,
        query_type=retrieval.query_type,
        was_rejected=not response_validation.valid,
        rejection_reason=(
            response_validation.reason if not response_validation.valid else None
        ),
    )

    return ChatCompletionResponse(
        id=completion_id,
        model=model_id,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=response_text),
            )
        ],
    )


_REFS_MARKER = "\n\n---\n**References:**"


def _references_suffix(processed: str) -> str:
    """Slice the References section off a postprocess_citations() result."""
    idx = processed.rfind(_REFS_MARKER)
    if idx == -1:
        return ""
    return processed[idx:]


def _stream_text(completion_id: str, model_id: str, text: str) -> StreamingResponse:
    """Stream a simple text response in OpenAI SSE format."""

    async def generate_events() -> AsyncIterator[str]:
        chunk = ChatCompletionChunk(
            id=completion_id,
            model=model_id,
            choices=[
                StreamChoice(
                    delta=DeltaMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_response(
    completion_id: str,
    model_id: str,
    conversation_history: list[dict],
    user_query: str,
    client_ip: str | None,
    start_time: float,
) -> StreamingResponse:
    """Stream LLM response in OpenAI SSE format with retrieval status."""

    async def generate_events() -> AsyncIterator[str]:
        # Initial chunk with role
        initial = ChatCompletionChunk(
            id=completion_id,
            model=model_id,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {initial.model_dump_json()}\n\n"

        # Show status while retrieving
        status_msg = "*Searching databases...*\n\n"
        status_chunk = ChatCompletionChunk(
            id=completion_id,
            model=model_id,
            choices=[StreamChoice(delta=DeltaMessage(content=status_msg))],
        )
        yield f"data: {status_chunk.model_dump_json()}\n\n"

        # Retrieve (this is the slow part - ~5-10s).
        # Run in executor to avoid blocking the event loop.
        loop = asyncio.get_event_loop()
        retrieval = await loop.run_in_executor(
            None, lambda: route_and_retrieve(user_query, k=5)
        )
        evidence = retrieval.clinvar_evidence
        llm_messages = build_augmented_messages(
            conversation_history, retrieval.prompt_context
        )

        accumulated_text = ""
        # The model is prompted to cite and some models append their own
        # "References:" block in-stream. We render the canonical block ourselves
        # afterwards (clickable links, correct titles), so we must NOT forward
        # the model's own block - otherwise the answer shows two References
        # sections (and tokens already shipped can't be recalled). Strategy:
        # forward tokens normally, but once a References heading appears in the
        # accumulated text, stop forwarding from the point it begins. We hold a
        # small tail buffer so a heading split across token boundaries is caught
        # before its first character is streamed.
        streamed_len = 0          # chars of accumulated_text already forwarded
        refs_started = False      # the model began its own References block
        TAIL = 24                 # >= longest heading prefix we match on

        def _refs_start(text: str) -> int | None:
            m = REFERENCES_HEADING.search(text)
            return m.start() if m else None

        try:
            async for token in generate_stream(llm_messages, model_id):
                accumulated_text += token
                if refs_started:
                    continue  # swallow the model's own ref block
                start = _refs_start(accumulated_text)
                if start is not None:
                    refs_started = True
                    safe_upto = start  # don't stream the heading or anything after
                else:
                    # Hold back a tail so a heading forming across tokens isn't
                    # streamed before we can detect it.
                    safe_upto = max(streamed_len, len(accumulated_text) - TAIL)
                if safe_upto > streamed_len:
                    out = accumulated_text[streamed_len:safe_upto]
                    streamed_len = safe_upto
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        model=model_id,
                        choices=[StreamChoice(delta=DeltaMessage(content=out))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            # Flush any held-back tail (only when the model never started a refs
            # block; if it did, everything from the heading on is intentionally
            # dropped).
            if not refs_started and streamed_len < len(accumulated_text):
                out = accumulated_text[streamed_len:]
                streamed_len = len(accumulated_text)
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    model=model_id,
                    choices=[StreamChoice(delta=DeltaMessage(content=out))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        except UnknownBackendError:
            error_chunk = ChatCompletionChunk(
                id=completion_id,
                model=model_id,
                choices=[
                    StreamChoice(
                        delta=DeltaMessage(content=f"\n\nUnknown model: {model_id!r}."),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            log_query(
                session_id=completion_id,
                user_id="openai-api",
                query=user_query,
                response="",
                evidence=evidence,
                model_used=model_id,
                latency_ms=latency_ms,
                client_ip=client_ip,
                query_type=retrieval.query_type,
                error_message=str(e),
            )
            error_chunk = ChatCompletionChunk(
                id=completion_id,
                model=model_id,
                choices=[
                    StreamChoice(
                        delta=DeltaMessage(content=f"\n\nError: {e}"),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Post-generation validation
        citations = retrieval.citations
        response_validation = validate_response(
            accumulated_text, evidence, citations=citations,
        )
        was_rejected = False
        if not response_validation.valid:
            accumulated_text = response_validation.filtered_response
            was_rejected = True

        # The body was already streamed token-by-token with bare [N] markers,
        # and the model's own References block (if any) was withheld mid-stream
        # above. We now append our canonical References block. postprocess_
        # citations() also strips any model-written ref block from its input, so
        # _references_suffix gives only our block - no duplication. (Rewriting
        # [N] -> [[N]](url) in the body in-place isn't possible after the tokens
        # have shipped.)
        processed = postprocess_citations(accumulated_text, citations)
        refs_suffix = _references_suffix(processed)
        if refs_suffix:
            refs_chunk = ChatCompletionChunk(
                id=completion_id,
                model=model_id,
                choices=[StreamChoice(delta=DeltaMessage(content=refs_suffix))],
            )
            yield f"data: {refs_chunk.model_dump_json()}\n\n"

        # Persist what the client actually received: the streamed body with the
        # model's own ref block stripped (matching what we forwarded), plus our
        # canonical block. status_msg is a transient UI frame and is excluded.
        streamed_body = REFERENCES_HEADING.split(accumulated_text)[0].rstrip()
        final_text = streamed_body + refs_suffix

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        log_query(
            session_id=completion_id,
            user_id="openai-api",
            query=user_query,
            response=final_text,
            evidence=evidence,
            model_used=model_id,
            latency_ms=latency_ms,
            client_ip=client_ip,
            query_type=retrieval.query_type,
            was_rejected=was_rejected,
            rejection_reason=(
                response_validation.reason if was_rejected else None
            ),
        )

        # Final chunk
        final = ChatCompletionChunk(
            id=completion_id,
            model=model_id,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        yield f"data: {final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
