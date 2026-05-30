"""
Shared embedding client for AttenGene.

AttenGene calls a remote embeddings server over HTTP. The server is
expected to host BAAI/bge-large-en-v1.5 (or any model that returns
1024-dim vectors) behind a TEI-shape /embed endpoint. The bundled
embeddings container in docker-compose.yml is one such server.

Set EMBEDDINGS_URL to the server's /embed URL. The function signatures
`encode(text) -> list[float]` and `encode_batch(texts) -> list[list[float]]`
are stable; callers don't see the transport.

For bulk indexing, EmbedInsertPipeline overlaps embed-side and
insert-side work across threads so the GPU behind the embeddings
server doesn't idle while pgvector is taking writes.
"""

import logging
import os
import queue
import threading
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "").strip()
EMBEDDINGS_TIMEOUT = float(os.getenv("EMBEDDINGS_TIMEOUT", "10.0"))
EMBEDDINGS_BATCH_SIZE = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32"))


class EmbeddingDimensionError(RuntimeError):
    """The configured embeddings backend returned the wrong vector length."""


def _require_url() -> str:
    if not EMBEDDINGS_URL:
        raise RuntimeError(
            "EMBEDDINGS_URL is not set. AttenGene needs an embeddings server "
            "(any TEI-shape /embed endpoint). The bundled embeddings "
            "container in docker-compose.yml provides one - set "
            "EMBEDDINGS_URL=http://embeddings:8081/embed, or point at your "
            "own server."
        )
    return EMBEDDINGS_URL


def _assert_dim(vec: list, source: str) -> None:
    if len(vec) != EMBEDDING_DIM:
        raise EmbeddingDimensionError(
            f"{source} returned {len(vec)}-dim vector; "
            f"expected {EMBEDDING_DIM} (BAAI/bge-large-en-v1.5). "
            f"Check that EMBEDDINGS_URL points at a server hosting "
            f"{EMBEDDING_MODEL!r} and not a different model."
        )


_dim_checked = False


def _check_dim_once(vec: list) -> None:
    global _dim_checked
    if _dim_checked:
        return
    _assert_dim(vec, f"embeddings server at {EMBEDDINGS_URL}")
    _dim_checked = True
    logger.info("Embeddings dimensionality check passed (%d) for %s",
                EMBEDDING_DIM, EMBEDDINGS_URL)


def _post_embed(payload: dict) -> list:
    url = _require_url()
    with httpx.Client(timeout=EMBEDDINGS_TIMEOUT) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    # TEI's /embed returns a bare list[list[float]]. Be liberal: also
    # accept {"embeddings": [...]} or {"data": [{"embedding": [...]}, ...]}.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "embeddings" in data:
            return data["embeddings"]
        if "data" in data and isinstance(data["data"], list):
            return [item["embedding"] for item in data["data"]]
    raise RuntimeError(
        f"Unrecognised embeddings response shape from {url}: "
        f"{type(data).__name__}"
    )


def encode(text: str) -> list[float]:
    """Encode one string to a normalized 1024-dim BGE-large embedding."""
    vecs = _post_embed({"inputs": [text]})
    if not vecs:
        raise RuntimeError(f"Empty embeddings response from {EMBEDDINGS_URL}")
    _check_dim_once(vecs[0])
    return vecs[0]


def encode_batch(texts: list[str]) -> list[list[float]]:
    """
    Encode a list of strings, preserving order.

    Returns one 1024-dim vector per input. Empty list in, empty list out.
    Chunks oversized inputs by EMBEDDINGS_BATCH_SIZE so a giant call
    doesn't OOM the server or trip its max-batch-tokens guardrail.
    """
    if not texts:
        return []
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDINGS_BATCH_SIZE):
        chunk = texts[start:start + EMBEDDINGS_BATCH_SIZE]
        vecs = _post_embed({"inputs": chunk})
        if len(vecs) != len(chunk):
            raise RuntimeError(
                f"Embeddings server returned {len(vecs)} vectors for "
                f"{len(chunk)} inputs"
            )
        _check_dim_once(vecs[0])
        out.extend(vecs)
    return out


# ---------- Indexer pipeline ----------


_PIPELINE_SENTINEL = object()


class EmbedInsertPipeline:
    """
    Overlap embed and insert work across threads for bulk indexing.

    The motivating problem: a serial `parse -> encode_batch -> INSERT` loop
    leaves the GPU idle while pgvector writes, and leaves pgvector idle
    while the embeddings server is encoding. Splitting the two stages
    across threads lets them run concurrently. Multiple embed workers also
    let a small RTX-class GPU stay above its idle clocks across a run.

    Two stages connected by bounded queues:

        producer (caller's parse loop)
          -> embed_queue [maxsize=queue_depth]
          -> N embed workers running `encode_batch(docs)`
          -> insert_queue [maxsize=queue_depth]
          -> 1 insert worker calling `sink(batch, embeddings)`

    The sink is the caller's existing per-table INSERT helper. It receives
    the original batch (so it can pull whatever non-text fields it needs)
    plus the parallel list of embeddings.

    Order is NOT preserved. With embed_workers > 1, batches can finish
    encoding in any order and reach the sink in any order. All three
    AttenGene indexers use `ON CONFLICT DO UPDATE`, which makes the final
    DB state deterministic regardless of insertion order, so this is fine.
    Do not use this pipeline for workloads that need ordered writes.

    Usage:

        def sink(batch: list[Section], embeddings: list[list[float]]) -> None:
            rows = [build_row(s, e) for s, e in zip(batch, embeddings)]
            insert_chunks(conn, rows)

        with EmbedInsertPipeline(sink=sink, embed_workers=2) as pipe:
            for batch, docs in iter_batches(...):
                pipe.submit(batch, docs)
        # __exit__ drains all queues and joins workers before returning.
        # Worker exceptions are re-raised on submit() or __exit__().

    Threading is safe here because httpx (embed worker) and psycopg2 (sink
    worker) both release the GIL on socket I/O.
    """

    def __init__(
        self,
        *,
        sink: Callable[[list, list[list[float]]], None],
        embed_workers: int = 2,
        queue_depth: int = 4,
    ) -> None:
        if embed_workers < 1:
            raise ValueError("embed_workers must be >= 1")
        if queue_depth < 1:
            raise ValueError("queue_depth must be >= 1")
        self._sink = sink
        self._embed_workers = embed_workers
        self._embed_queue: "queue.Queue" = queue.Queue(maxsize=queue_depth)
        self._insert_queue: "queue.Queue" = queue.Queue(maxsize=queue_depth)
        self._threads: list[threading.Thread] = []
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        self._started = False
        self._closed = False

    def __enter__(self) -> "EmbedInsertPipeline":
        for i in range(self._embed_workers):
            t = threading.Thread(
                target=self._embed_loop,
                name=f"embed-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        t = threading.Thread(
            target=self._insert_loop,
            name="insert",
            daemon=True,
        )
        t.start()
        self._threads.append(t)
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def submit(self, batch: list, docs: list[str]) -> None:
        if not self._started or self._closed:
            raise RuntimeError("EmbedInsertPipeline is not active")
        self._reraise_if_failed()
        if len(batch) != len(docs):
            raise ValueError(
                f"batch length {len(batch)} != docs length {len(docs)}"
            )
        self._embed_queue.put((batch, docs))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in range(self._embed_workers):
            self._embed_queue.put(_PIPELINE_SENTINEL)
        for t in self._threads:
            t.join()
        self._reraise_if_failed()

    def _embed_loop(self) -> None:
        try:
            while True:
                item = self._embed_queue.get()
                if item is _PIPELINE_SENTINEL:
                    return
                if self._error is not None:
                    continue
                batch, docs = item
                embeddings = encode_batch(docs)
                self._insert_queue.put((batch, embeddings))
        except BaseException as e:
            self._record_error(e)
        finally:
            try:
                self._insert_queue.put_nowait(_PIPELINE_SENTINEL)
            except queue.Full:
                pass

    def _insert_loop(self) -> None:
        sentinels_seen = 0
        try:
            while sentinels_seen < self._embed_workers:
                try:
                    item = self._insert_queue.get(timeout=0.2)
                except queue.Empty:
                    if self._error is not None and self._closed:
                        return
                    continue
                if item is _PIPELINE_SENTINEL:
                    sentinels_seen += 1
                    continue
                if self._error is not None:
                    continue
                batch, embeddings = item
                self._sink(batch, embeddings)
        except BaseException as e:
            self._record_error(e)

    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _reraise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error
