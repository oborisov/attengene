"""
Async LLM client with multi-backend dispatch.

The RAG pipeline upstream is identical across backends; only the final
HTTP call differs. Each backend is exposed to OWUI as a separate model id
on /v1/models (attengene-local, attengene-mistral, later attengene-claude),
selected per-request via the `model` field.
"""

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


@dataclass(frozen=True)
class Backend:
    """One generation backend, exposed as a single OWUI model id."""

    model_id: str          # what OWUI sees and what /v1/models advertises
    base_url: str          # full chat-completions URL
    api_key: str | None    # None = no Authorization header (llama-server)
    upstream_model: str    # value sent in the request body's `model` field


def _build_backends() -> dict[str, Backend]:
    """Read env vars, drop backends whose required config is missing."""
    out: dict[str, Backend] = {}

    # BACKEND_LOCAL_KEY is optional: empty/unset = no Authorization header
    # (e.g. dev against a no-auth Ollama/llama-server). When set, every
    # outbound call to the local backend carries `Authorization: Bearer <key>`,
    # for an upstream llama-server started with --api-key. _build_headers()
    # already no-ops on a falsy key, so `or None` keeps the no-auth path
    # identical.
    out["attengene-local"] = Backend(
        model_id="attengene-local",
        base_url=os.getenv(
            "BACKEND_LOCAL_URL", "http://localhost:8080/v1/chat/completions"
        ),
        api_key=os.getenv("BACKEND_LOCAL_KEY", "").strip() or None,
        upstream_model=os.getenv("BACKEND_LOCAL_MODEL", "auto"),
    )

    mistral_key = os.getenv("BACKEND_MISTRAL_KEY", "").strip()
    if mistral_key:
        out["attengene-mistral"] = Backend(
            model_id="attengene-mistral",
            base_url=os.getenv(
                "BACKEND_MISTRAL_URL",
                "https://api.mistral.ai/v1/chat/completions",
            ),
            api_key=mistral_key,
            upstream_model=os.getenv(
                "BACKEND_MISTRAL_MODEL", "mistral-large-latest"
            ),
        )

    return out


BACKENDS: dict[str, Backend] = _build_backends()


class UnknownBackendError(KeyError):
    """Raised when /v1/chat/completions is called with an unrecognised model id."""


def get_backend(model_id: str) -> Backend:
    """Look up a backend, raise UnknownBackendError if not registered."""
    if model_id not in BACKENDS:
        raise UnknownBackendError(model_id)
    return BACKENDS[model_id]


def available_model_ids() -> list[str]:
    """Backends that survived env-var validation, in display order."""
    return sorted(BACKENDS.keys())


async def resolve_local_upstream_model() -> None:
    """
    If BACKEND_LOCAL_MODEL is `auto`, query the local server's /v1/models
    once at startup and cache the first id. No-op for explicit settings.
    Failures fall back to a sentinel string so the request still goes out.
    """
    local = BACKENDS.get("attengene-local")
    if local is None or local.upstream_model != "auto":
        return

    models_url = local.base_url.rsplit("/chat/completions", 1)[0] + "/models"
    try:
        async with httpx.AsyncClient() as client:
            # Carry the same bearer as chat calls - llama-server with
            # --api-key returns 401 on /v1/models too, which would otherwise
            # silently drop us to the "local" sentinel below.
            response = await client.get(
                models_url, headers=_build_headers(local), timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            resolved = data["data"][0]["id"]
    except Exception as e:
        logger.warning(
            "Could not resolve local upstream model from %s: %s. "
            "Using sentinel 'local'.", models_url, e
        )
        resolved = "local"

    BACKENDS["attengene-local"] = Backend(
        model_id=local.model_id,
        base_url=local.base_url,
        api_key=local.api_key,
        upstream_model=resolved,
    )
    logger.info("Resolved attengene-local upstream model: %s", resolved)


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    response_text: str
    model_used: str


def _build_payload(backend: Backend, messages: list[dict[str, str]], *, stream: bool) -> dict:
    return {
        "model": backend.upstream_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": stream,
    }


def _build_headers(backend: Backend) -> dict[str, str]:
    if backend.api_key:
        return {"Authorization": f"Bearer {backend.api_key}"}
    return {}


def _strip_think_blocks(text: str) -> str:
    """Scrub Qwen3 <think>...</think> blocks from a complete response."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


async def generate(messages: list[dict[str, str]], model_id: str) -> LLMResponse:
    """
    Generate a complete (non-streaming) response from `model_id`.

    Raises:
        UnknownBackendError: model_id is not in BACKENDS.
        RuntimeError: upstream HTTP failure.
    """
    backend = get_backend(model_id)
    payload = _build_payload(backend, messages, stream=False)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                backend.base_url,
                json=payload,
                headers=_build_headers(backend),
                timeout=LLM_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            content = _strip_think_blocks(content)
            return LLMResponse(response_text=content, model_used=model_id)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to LLM backend {model_id} at {backend.base_url}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"LLM backend {model_id} timed out after {LLM_TIMEOUT}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM backend {model_id} error: {e.response.status_code}"
            ) from e


async def generate_stream(
    messages: list[dict[str, str]], model_id: str
) -> AsyncIterator[str]:
    """
    Stream tokens from `model_id`, filtering <think>...</think> blocks
    incrementally.

    Raises:
        UnknownBackendError: model_id is not in BACKENDS.
        RuntimeError: upstream HTTP failure.
    """
    backend = get_backend(model_id)
    payload = _build_payload(backend, messages, stream=True)

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                backend.base_url,
                json=payload,
                headers=_build_headers(backend),
                timeout=LLM_TIMEOUT,
            ) as response:
                response.raise_for_status()
                in_think = False

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue

                        if not in_think and "<think>" in content:
                            in_think = True
                            before = content.split("<think>")[0]
                            if before:
                                yield before
                            continue
                        if in_think:
                            if "</think>" in content:
                                in_think = False
                                after = content.split("</think>", 1)[1]
                                if after:
                                    yield after
                            continue
                        yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to LLM backend {model_id} at {backend.base_url}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"LLM backend {model_id} timed out after {LLM_TIMEOUT}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM backend {model_id} error: {e.response.status_code}"
            ) from e


if __name__ == "__main__":
    import asyncio

    async def test():
        print(f"Registered backends: {available_model_ids()}")
        await resolve_local_upstream_model()
        print(f"After resolve: {BACKENDS}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ]

        print("\nTesting non-streaming against attengene-local...")
        result = await generate(messages, "attengene-local")
        print(f"Response: {result.response_text}")
        print(f"Model: {result.model_used}")

        print("\nTesting streaming...")
        async for chunk in generate_stream(messages, "attengene-local"):
            print(chunk, end="", flush=True)
        print()

    asyncio.run(test())
