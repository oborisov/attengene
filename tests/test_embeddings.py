"""
Unit tests for app/embeddings.py: HTTP transport, batching, dimensionality.

Run with: python -m unittest tests.test_embeddings
"""

import importlib
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _embeddings_env(**env: str):
    """
    Reload app.embeddings with a clean env so module-level constants
    (EMBEDDINGS_URL, EMBEDDINGS_BATCH_SIZE, ...) reflect the test config.
    Also resets the first-call dimensionality flag.
    """
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("EMBEDDINGS_")}
    base.update(env)
    with patch.dict(os.environ, base, clear=True):
        from app import embeddings
        importlib.reload(embeddings)
        yield embeddings


class _FakeResponse:
    def __init__(self, payload, status: int = 200):
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


class _RecordingClient:
    def __init__(self, *args, response=None, **kwargs):
        self.calls = []
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json):
        self.calls.append({"url": url, "json": json})
        return self._response


def _vec(value: float = 0.1, dim: int = 1024) -> list[float]:
    return [value] * dim


class RequiredUrlTest(unittest.TestCase):
    def test_missing_url_raises_with_clear_message(self):
        with _embeddings_env() as emb:
            with self.assertRaises(RuntimeError) as ctx:
                emb.encode("hi")
            msg = str(ctx.exception)
            self.assertIn("EMBEDDINGS_URL", msg)

    def test_url_set_sends_request(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            response = _FakeResponse([_vec()])
            captured = []

            def factory(*args, **kwargs):
                c = _RecordingClient(response=response)
                captured.append(c)
                return c

            with patch.object(emb.httpx, "Client", factory):
                vec = emb.encode("hello")

            self.assertEqual(len(vec), emb.EMBEDDING_DIM)
            self.assertEqual(captured[0].calls[0]["url"],
                             "http://embeddings.test/embed")
            self.assertEqual(captured[0].calls[0]["json"], {"inputs": ["hello"]})


class DimensionalityCheckTest(unittest.TestCase):
    def test_wrong_dim_raises_with_clear_message(self):
        with _embeddings_env(EMBEDDINGS_URL="http://wrong.test/embed") as emb:
            response = _FakeResponse([_vec(dim=768)])

            def factory(*args, **kwargs):
                return _RecordingClient(response=response)

            with patch.object(emb.httpx, "Client", factory):
                with self.assertRaises(emb.EmbeddingDimensionError) as ctx:
                    emb.encode("hello")

            msg = str(ctx.exception)
            self.assertIn("768", msg)
            self.assertIn("1024", msg)
            self.assertIn("bge-large", msg)

    def test_dim_check_runs_once_then_caches(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            response = _FakeResponse([_vec()])

            def factory(*args, **kwargs):
                return _RecordingClient(response=response)

            with patch.object(emb.httpx, "Client", factory):
                emb.encode("first")
                self.assertTrue(emb._dim_checked)
                emb.encode("second")


class BatchingTest(unittest.TestCase):
    def test_batch_size_chunks_oversized_input(self):
        with _embeddings_env(
            EMBEDDINGS_URL="http://embeddings.test/embed",
            EMBEDDINGS_BATCH_SIZE="3",
        ) as emb:
            captured = []

            def factory(*args, **kwargs):
                c = _RecordingClient()
                _orig_post = c.post

                def post(url, *, json):
                    _orig_post(url, json=json)
                    return _FakeResponse([_vec() for _ in json["inputs"]])

                c.post = post
                captured.append(c)
                return c

            with patch.object(emb.httpx, "Client", factory):
                out = emb.encode_batch([f"text-{i}" for i in range(7)])

            self.assertEqual(len(out), 7)
            for vec in out:
                self.assertEqual(len(vec), emb.EMBEDDING_DIM)
            self.assertEqual(len(captured), 3)
            self.assertEqual(len(captured[0].calls[0]["json"]["inputs"]), 3)
            self.assertEqual(len(captured[1].calls[0]["json"]["inputs"]), 3)
            self.assertEqual(len(captured[2].calls[0]["json"]["inputs"]), 1)

    def test_empty_input_returns_empty(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            def factory(*args, **kwargs):
                raise AssertionError("HTTP client constructed for empty input")

            with patch.object(emb.httpx, "Client", factory):
                self.assertEqual(emb.encode_batch([]), [])


class ResponseShapesTest(unittest.TestCase):
    """TEI's /embed returns a bare list; tolerate the other common shapes too."""

    def _call_with_payload(self, payload):
        with _embeddings_env(EMBEDDINGS_URL="http://x.test/embed") as emb:
            response = _FakeResponse(payload)

            def factory(*args, **kwargs):
                return _RecordingClient(response=response)

            with patch.object(emb.httpx, "Client", factory):
                return emb.encode("hi")

    def test_bare_list_shape(self):
        vec = self._call_with_payload([_vec()])
        self.assertEqual(len(vec), 1024)

    def test_embeddings_key_shape(self):
        vec = self._call_with_payload({"embeddings": [_vec()]})
        self.assertEqual(len(vec), 1024)

    def test_openai_data_shape(self):
        vec = self._call_with_payload({"data": [{"embedding": _vec(), "index": 0}]})
        self.assertEqual(len(vec), 1024)


class EmbedInsertPipelineTest(unittest.TestCase):
    """
    EmbedInsertPipeline overlaps embed and insert work across threads.

    Tests mock encode_batch so no real HTTP calls happen. The sink callable
    is recorded so we can assert what reached the insert stage.
    """

    def test_round_trip_all_batches_reach_sink(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            seen_inputs: list[list[str]] = []

            def fake_encode(texts):
                seen_inputs.append(list(texts))
                return [[float(i)] * emb.EMBEDDING_DIM for i in range(len(texts))]

            sink_calls: list[tuple] = []

            def sink(batch, embeddings):
                sink_calls.append((batch, embeddings))

            with patch.object(emb, "encode_batch", fake_encode):
                with emb.EmbedInsertPipeline(sink=sink, embed_workers=2) as pipe:
                    for i in range(5):
                        batch = [{"id": i, "doc": f"text-{i}"}]
                        pipe.submit(batch, [b["doc"] for b in batch])

            self.assertEqual(len(sink_calls), 5)
            ids = sorted(call[0][0]["id"] for call in sink_calls)
            self.assertEqual(ids, [0, 1, 2, 3, 4])
            for batch, embeddings in sink_calls:
                self.assertEqual(len(batch), len(embeddings))
                self.assertEqual(len(embeddings[0]), emb.EMBEDDING_DIM)

    def test_embed_error_propagates_on_close(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            def broken_encode(texts):
                raise RuntimeError("embeddings server down")

            def sink(batch, embeddings):
                self.fail("sink should not be called when embed fails")

            with self.assertRaises(RuntimeError) as ctx:
                with patch.object(emb, "encode_batch", broken_encode):
                    with emb.EmbedInsertPipeline(sink=sink) as pipe:
                        for i in range(3):
                            try:
                                pipe.submit([{"i": i}], ["text"])
                            except RuntimeError:
                                raise

            self.assertIn("embeddings server down", str(ctx.exception))

    def test_sink_error_propagates_on_close(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            def fake_encode(texts):
                return [[0.1] * emb.EMBEDDING_DIM for _ in texts]

            def broken_sink(batch, embeddings):
                raise RuntimeError("DB unreachable")

            with self.assertRaises(RuntimeError) as ctx:
                with patch.object(emb, "encode_batch", fake_encode):
                    with emb.EmbedInsertPipeline(sink=broken_sink) as pipe:
                        pipe.submit([{"i": 0}], ["text"])

            self.assertIn("DB unreachable", str(ctx.exception))

    def test_batch_docs_length_mismatch_raises(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            def fake_encode(texts):
                return [[0.0] * emb.EMBEDDING_DIM for _ in texts]

            def sink(batch, embeddings):
                pass

            with patch.object(emb, "encode_batch", fake_encode):
                with emb.EmbedInsertPipeline(sink=sink) as pipe:
                    with self.assertRaises(ValueError):
                        pipe.submit([{"a": 1}, {"a": 2}], ["one-doc-for-two"])

    def test_invalid_worker_count_raises(self):
        with _embeddings_env(EMBEDDINGS_URL="http://embeddings.test/embed") as emb:
            def sink(batch, embeddings):
                pass

            with self.assertRaises(ValueError):
                emb.EmbedInsertPipeline(sink=sink, embed_workers=0)
            with self.assertRaises(ValueError):
                emb.EmbedInsertPipeline(sink=sink, queue_depth=0)


if __name__ == "__main__":
    unittest.main()
