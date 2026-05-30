# API

AttenGene exposes an OpenAI-compatible HTTP surface (`/v1/*`) plus a
`/health` liveness probe. The OpenAI surface is the canonical entry
point - any OpenAI-API client (Open WebUI, LibreChat, your own app)
works without code changes.

Auth: when `ATTENGENE_API_KEY` is set, `/v1/*` requires
`Authorization: Bearer <key>`.

## `GET /health`

Liveness check. Always returns `{"status": "ok"}`.

```bash
curl http://localhost:8000/health
```

## `GET /v1/models`

Lists every configured generation backend as an OpenAI model id.

```json
{
  "object": "list",
  "data": [
    {
      "id": "attengene-local",
      "object": "model",
      "created": 1747000000,
      "owned_by": "attengene",
      "description": "AttenGene RAG with local llama-server backend (GDPR-safe)."
    }
  ]
}
```

Backends not present in the response are not active. `attengene-mistral`
only appears when `BACKEND_MISTRAL_KEY` is set.

## `POST /v1/chat/completions`

Standard OpenAI request shape. The `model` field picks which backend
runs the final LLM call - the rest of the RAG pipeline (retrieval,
guardrails, citation assembly, audit) is identical across backends.

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
      "model": "attengene-local",
      "messages": [
        {"role": "user", "content": "What is known about BRCA1 c.5266dupC?"}
      ]
    }'
```

Streaming via `"stream": true` returns SSE chunks in the OpenAI
`chat.completion.chunk` shape. Citations are appended to the final
chunk.

## Error shapes

```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```
