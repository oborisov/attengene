"""
Unit tests for app/embeddings_server.py.

We mock out the model load and the encode call so the tests do not require
CUDA, the sentence-transformers wheel, or the 1.3 GB of model weights. The
TestClient drives the FastAPI app through its normal lifespan, so the
/health 503->200 transition is exercised too.

Run with: python -m unittest tests.test_embeddings_server
"""

import importlib
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _server_env(**env: str):
    """
    Reload app.embeddings_server with a clean env so module-level constants
    (EMBEDDINGS_BATCH_SIZE, EMBEDDINGS_MODEL, ...) reflect the test config.
    """
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("EMBEDDINGS_")}
    base.update(env)
    with patch.dict(os.environ, base, clear=True):
        from app import embeddings_server
        importlib.reload(embeddings_server)
        yield embeddings_server


class _FakeModel:
    """
    Stand-in for SentenceTransformer. Records every encode call and returns
    a deterministic vector per input (first element = input index, rest 0).
    """

    DIM = 1024

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, *, normalize_embeddings, show_progress_bar,
               convert_to_numpy):
        # texts is a list[str]; record it so tests can assert chunking.
        self.calls.append(list(texts))
        import numpy as np
        return np.array(
            [[float(i)] + [0.0] * (self.DIM - 1) for i in range(len(texts))],
            dtype=np.float32,
        )


def _client_with_fake_model(server_module):
    """
    Spin up a TestClient that uses _FakeModel in place of _load_model().
    Returns (client_context_manager, fake_model_instance).
    """
    from fastapi.testclient import TestClient
    fake = _FakeModel()
    # _load_model runs inside the lifespan hook; patch it before the client
    # context opens so the hook installs our fake.
    patched = patch.object(server_module, "_load_model", lambda: fake)
    patched.start()
    client = TestClient(server_module.app)
    return client, fake, patched


class HealthTest(unittest.TestCase):
    def test_health_200_after_lifespan_load(self):
        with _server_env() as srv:
            client, _fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    r = client.get("/health")
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(r.json(), {"status": "ok"})
            finally:
                patched.stop()

    def test_health_503_after_lifespan_exit(self):
        # After the TestClient context exits, the lifespan tears down and
        # _model_ready flips back to False. A fresh GET without re-entering
        # the context must return 503.
        with _server_env() as srv:
            client, _fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    pass  # lifespan startup + shutdown
                r = client.get("/health")
                self.assertEqual(r.status_code, 503)
                self.assertEqual(r.json(), {"status": "loading"})
            finally:
                patched.stop()


class EmbedEndpointTest(unittest.TestCase):
    def test_embed_returns_bare_list(self):
        with _server_env() as srv:
            client, fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    r = client.post("/embed", json={"inputs": ["a", "b", "c"]})
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertIsInstance(body, list)
                self.assertEqual(len(body), 3)
                for vec in body:
                    self.assertEqual(len(vec), _FakeModel.DIM)
                self.assertEqual(fake.calls, [["a", "b", "c"]])
            finally:
                patched.stop()

    def test_embed_chunks_oversized_input(self):
        with _server_env(EMBEDDINGS_BATCH_SIZE="2") as srv:
            client, fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    r = client.post(
                        "/embed", json={"inputs": ["a", "b", "c", "d", "e"]},
                    )
                self.assertEqual(r.status_code, 200)
                self.assertEqual(len(r.json()), 5)
                # 5 inputs at batch size 2 -> 3 encode() calls (2, 2, 1).
                self.assertEqual(len(fake.calls), 3)
                self.assertEqual(fake.calls[0], ["a", "b"])
                self.assertEqual(fake.calls[1], ["c", "d"])
                self.assertEqual(fake.calls[2], ["e"])
            finally:
                patched.stop()

    def test_embed_rejects_empty_inputs(self):
        with _server_env() as srv:
            client, _fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    r = client.post("/embed", json={"inputs": []})
                # Pydantic min_length=1 -> 422.
                self.assertEqual(r.status_code, 422)
            finally:
                patched.stop()


class OpenAIEmbeddingsEndpointTest(unittest.TestCase):
    def test_openai_shape_with_list_input(self):
        with _server_env() as srv:
            client, _fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    r = client.post(
                        "/v1/embeddings",
                        json={"input": ["alpha", "beta"], "model": "bge-large"},
                    )
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertEqual(body["object"], "list")
                self.assertEqual(body["model"], "bge-large")
                self.assertEqual(len(body["data"]), 2)
                self.assertEqual(body["data"][0]["object"], "embedding")
                self.assertEqual(body["data"][0]["index"], 0)
                self.assertEqual(body["data"][1]["index"], 1)
                self.assertEqual(len(body["data"][0]["embedding"]),
                                 _FakeModel.DIM)
                self.assertIn("usage", body)
            finally:
                patched.stop()

    def test_openai_shape_with_string_input(self):
        # OpenAI's spec accepts a single string too.
        with _server_env() as srv:
            client, _fake, patched = _client_with_fake_model(srv)
            try:
                with client:
                    r = client.post(
                        "/v1/embeddings",
                        json={"input": "just one string"},
                    )
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertEqual(len(body["data"]), 1)
                # No model in request -> falls back to configured model.
                self.assertEqual(body["model"], srv.EMBEDDINGS_MODEL)
            finally:
                patched.stop()


class LoaderTest(unittest.TestCase):
    """
    Direct tests on _load_model; do not go through the FastAPI lifespan.
    """

    def test_cpu_device_is_rejected(self):
        with _server_env(EMBEDDINGS_DEVICE="cpu") as srv:
            with self.assertRaises(RuntimeError) as ctx:
                srv._load_model()
            self.assertIn("cuda", str(ctx.exception).lower())

    def test_invalid_dtype_raises(self):
        with _server_env(EMBEDDINGS_DTYPE="quantum") as srv:
            # _torch_dtype is the gate; check it directly so we don't need a
            # real CUDA build to exercise the failure path.
            with self.assertRaises(ValueError) as ctx:
                srv._torch_dtype("quantum")
            self.assertIn("quantum", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
