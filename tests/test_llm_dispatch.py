"""
Unit tests for app/llm.py multi-backend dispatch.

Run with: python -m unittest tests.test_llm_dispatch
"""

import asyncio
import importlib
import json
import os
import unittest
from collections.abc import AsyncIterator
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _backends_env(**env: str):
    """
    Reload app.llm with a clean env, so BACKENDS reflects exactly what we set.
    Strips any BACKEND_* / LLM_* keys the host shell may have leaking in.
    """
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("BACKEND_") and not k.startswith("LLM_")}
    base.update(env)
    with patch.dict(os.environ, base, clear=True):
        from app import llm
        importlib.reload(llm)
        try:
            yield llm
        finally:
            pass


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                "fake", request=None, response=self,  # type: ignore[arg-type]
            )

    def json(self):
        return self._payload


class _FakeStream:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        pass

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RecordingClient:
    """Minimal httpx.AsyncClient stand-in that records outbound requests."""

    def __init__(self, *args, response: _FakeResponse | None = None,
                 stream_lines: list[str] | None = None, **kwargs):
        self.calls: list[dict] = []
        self._response = response
        self._stream_lines = stream_lines or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, json, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers or {}})
        assert self._response is not None
        return self._response

    def stream(self, method, url, *, json, headers=None, timeout=None):
        self.calls.append({
            "method": method, "url": url, "json": json, "headers": headers or {},
        })
        return _FakeStream(self._stream_lines)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


class AvailableModelIdsTest(unittest.TestCase):
    def test_only_local_when_mistral_key_missing(self):
        with _backends_env() as llm:
            self.assertEqual(llm.available_model_ids(), ["attengene-local"])

    def test_mistral_appears_when_key_present(self):
        with _backends_env(BACKEND_MISTRAL_KEY="sk-test") as llm:
            self.assertEqual(
                llm.available_model_ids(),
                ["attengene-local", "attengene-mistral"],
            )

    def test_empty_mistral_key_is_dropped(self):
        with _backends_env(BACKEND_MISTRAL_KEY="   ") as llm:
            self.assertEqual(llm.available_model_ids(), ["attengene-local"])

    def test_unknown_backend_raises(self):
        with _backends_env() as llm:
            with self.assertRaises(llm.UnknownBackendError):
                llm.get_backend("attengene-claude")


class GenerateDispatchTest(unittest.TestCase):
    def _patch_client(self, llm_mod, *, response=None, stream_lines=None):
        captured: list[_RecordingClient] = []

        def factory(*args, **kwargs):
            client = _RecordingClient(
                response=response, stream_lines=stream_lines,
            )
            captured.append(client)
            return client

        return patch.object(llm_mod.httpx, "AsyncClient", factory), captured

    def test_local_dispatch_no_auth_header(self):
        with _backends_env(
            BACKEND_LOCAL_URL="http://local.test/v1/chat/completions",
            BACKEND_LOCAL_MODEL="pinned-local-model",
        ) as llm:
            response = _FakeResponse(
                {"choices": [{"message": {"content": "hi"}}]}
            )
            patcher, captured = self._patch_client(llm, response=response)
            with patcher:
                result = _run(llm.generate([{"role": "user", "content": "q"}],
                                            "attengene-local"))

            self.assertEqual(result.response_text, "hi")
            self.assertEqual(result.model_used, "attengene-local")
            call = captured[0].calls[0]
            self.assertEqual(call["url"], "http://local.test/v1/chat/completions")
            self.assertNotIn("Authorization", call["headers"])
            self.assertEqual(call["json"]["model"], "pinned-local-model")
            # `think` field must not appear in the outbound payload anymore.
            self.assertNotIn("think", call["json"])

    def test_mistral_dispatch_bearer_header_and_distinct_url(self):
        with _backends_env(
            BACKEND_MISTRAL_KEY="sk-secret",
            BACKEND_MISTRAL_URL="https://mistral.test/v1/chat/completions",
            BACKEND_MISTRAL_MODEL="mistral-medium",
        ) as llm:
            response = _FakeResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            patcher, captured = self._patch_client(llm, response=response)
            with patcher:
                _run(llm.generate([{"role": "user", "content": "q"}],
                                  "attengene-mistral"))

            call = captured[0].calls[0]
            self.assertEqual(call["url"], "https://mistral.test/v1/chat/completions")
            self.assertEqual(call["headers"]["Authorization"], "Bearer sk-secret")
            self.assertEqual(call["json"]["model"], "mistral-medium")

    def test_unknown_model_raises_unknown_backend_error(self):
        with _backends_env() as llm:
            with self.assertRaises(llm.UnknownBackendError):
                _run(llm.generate([{"role": "user", "content": "q"}],
                                  "attengene-claude"))

    def test_think_blocks_stripped_from_complete_response(self):
        with _backends_env() as llm:
            response = _FakeResponse({"choices": [{"message": {
                "content": "<think>internal</think>final answer",
            }}]})
            patcher, _ = self._patch_client(llm, response=response)
            with patcher:
                result = _run(llm.generate([{"role": "user", "content": "q"}],
                                            "attengene-local"))
            self.assertEqual(result.response_text, "final answer")


class GenerateStreamDispatchTest(unittest.TestCase):
    def _patch_client(self, llm_mod, stream_lines):
        captured: list[_RecordingClient] = []

        def factory(*args, **kwargs):
            client = _RecordingClient(stream_lines=stream_lines)
            captured.append(client)
            return client

        return patch.object(llm_mod.httpx, "AsyncClient", factory), captured

    @staticmethod
    def _sse(content: str) -> str:
        return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})

    def test_stream_passes_model_to_outbound_payload(self):
        with _backends_env(
            BACKEND_MISTRAL_KEY="sk-secret",
            BACKEND_MISTRAL_MODEL="mistral-medium",
        ) as llm:
            lines = [self._sse("hello "), self._sse("world"), "data: [DONE]"]
            patcher, captured = self._patch_client(llm, lines)
            with patcher:
                async def collect():
                    out: list[str] = []
                    async for token in llm.generate_stream(
                        [{"role": "user", "content": "q"}], "attengene-mistral"
                    ):
                        out.append(token)
                    return out
                tokens = _run(collect())

            self.assertEqual(tokens, ["hello ", "world"])
            call = captured[0].calls[0]
            self.assertEqual(call["json"]["model"], "mistral-medium")
            self.assertEqual(call["headers"]["Authorization"], "Bearer sk-secret")

    def test_stream_filters_think_blocks_across_chunks(self):
        with _backends_env() as llm:
            lines = [
                self._sse("intro "),
                self._sse("<think>hidden"),
                self._sse(" more hidden</think>"),
                self._sse("conclusion"),
                "data: [DONE]",
            ]
            patcher, _ = self._patch_client(llm, lines)
            with patcher:
                async def collect():
                    out: list[str] = []
                    async for token in llm.generate_stream(
                        [{"role": "user", "content": "q"}], "attengene-local"
                    ):
                        out.append(token)
                    return out
                tokens = _run(collect())
            self.assertEqual("".join(tokens), "intro conclusion")


if __name__ == "__main__":
    unittest.main()
