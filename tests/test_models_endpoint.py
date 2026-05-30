"""
Unit tests for the /v1/models endpoint filtering on registered backends.

Run with: python -m unittest tests.test_models_endpoint
"""

import importlib
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _isolated_env(**env: str):
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("BACKEND_") and not k.startswith("LLM_")
            and k != "ATTENGENE_API_KEY"}
    base.update(env)
    with patch.dict(os.environ, base, clear=True):
        # Reload llm so BACKENDS reflects this env, then auth + routes that
        # import llm.available_model_ids at call time pick up the new state.
        from app import llm
        importlib.reload(llm)
        from app import auth
        importlib.reload(auth)
        from app import routes_openai
        importlib.reload(routes_openai)
        yield routes_openai


def _client(routes_module):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(routes_module.router)
    return TestClient(app)


class ModelsEndpointTest(unittest.TestCase):
    def test_local_only_when_no_mistral_key(self):
        with _isolated_env() as routes:
            client = _client(routes)
            response = client.get("/v1/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            ids = [m["id"] for m in data["data"]]
            self.assertEqual(ids, ["attengene-local"])

    def test_both_backends_when_mistral_key_set(self):
        with _isolated_env(BACKEND_MISTRAL_KEY="sk-test") as routes:
            client = _client(routes)
            response = client.get("/v1/models")
            self.assertEqual(response.status_code, 200)
            ids = [m["id"] for m in response.json()["data"]]
            self.assertEqual(ids, ["attengene-local", "attengene-mistral"])

    def test_mistral_entry_has_non_clinical_warning(self):
        with _isolated_env(BACKEND_MISTRAL_KEY="sk-test") as routes:
            client = _client(routes)
            entries = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
            self.assertIn("Non-clinical", entries["attengene-mistral"]["description"])
            self.assertIn("GDPR-safe", entries["attengene-local"]["description"])


class UnknownModelErrorTest(unittest.TestCase):
    def test_chat_completions_with_unknown_model_returns_400(self):
        with _isolated_env() as routes:
            client = _client(routes)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "attengene-claude",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(response.status_code, 400)
            body = response.json()
            self.assertEqual(body["error"]["type"], "invalid_request_error")
            self.assertEqual(body["error"]["param"], "model")
            self.assertIn("attengene-claude", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
