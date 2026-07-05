"""
Cognify core — the ECL orchestration (Extract -> Cognify -> Load), the shared
embedder, and the Backend protocol both backends implement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from . import config, extractor as _ex
from .loader import Document, load

log = logging.getLogger("cognify")
_model = None


def get_model():
    global _model
    if _model is None:
        if config.EMBED_PROVIDER == "fastembed":
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                raise RuntimeError("COGNIFY_EMBED_PROVIDER=fastembed but fastembed is not "
                                   "installed. Run: pip install 'cognify-kg[fastembed]'") from e
            name = config.EMBED_MODEL if "/" in config.EMBED_MODEL \
                else f"sentence-transformers/{config.EMBED_MODEL}"
            _model = TextEmbedding(model_name=name)
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("sentence-transformers is not installed. Run: pip install "
                                   "'cognify-kg[st]' or set COGNIFY_EMBED_PROVIDER=fastembed") from e
            _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    m = get_model()
    if config.EMBED_PROVIDER == "fastembed":
        v = np.asarray(list(m.embed(texts, batch_size=64)), dtype=np.float32)
        # guarantee the shared L2-normalized space regardless of provider defaults
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    v = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def _embed_texts(backend, texts: list[str]) -> np.ndarray:
    """Use the backend's own embedder if it has one (client boxes use a torch-free
    ONNX MiniLM via ChromaDB), else the shared sentence-transformers model."""
    fn = getattr(backend, "embed_texts", None)
    return fn(texts) if callable(fn) else embed(texts)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    title: str
    tenant: str
    namespace: str
    chunks: int
    entities: int
    relations: int
    extracted: bool


@dataclass(frozen=True)
class RecallResult:
    query: str
    tenant: str
    chunks: tuple[dict, ...]
    entities: tuple[dict, ...]
    relations: tuple[dict, ...]


class Backend(Protocol):
    def load_document(self, doc: Document, *, tenant: str, namespace: str, agent: str,
                      chunk_vecs: np.ndarray, extractions: dict) -> None: ...
    def search(self, qvec: np.ndarray, *, tenant: str, namespace: Optional[str], k: int) -> list[dict]: ...
    def expand(self, chunk_ids: list[str], *, tenant: str, hops: int) -> dict: ...
    def delete_document(self, doc_id: str, *, tenant: str) -> dict: ...
    def stats(self, *, tenant: Optional[str], namespace: Optional[str] = None) -> dict: ...


def _extract_one(chunk) -> tuple[str, "_ex.Extraction"]:
    try:
        return chunk.id, _ex.extract(chunk.text)
    except Exception as e:
        log.warning("extraction failed for %s: %s", chunk.id, e)
        return chunk.id, _ex.Extraction()


def _extract_all(chunks, workers: int) -> dict:
    if workers <= 1 or len(chunks) <= 1:
        return dict(_extract_one(c) for c in chunks)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
        return dict(pool.map(_extract_one, chunks))


def ingest(backend, path_or_text: str, *, tenant: str = "default", namespace: str = "default",
           agent: str = "agent", is_path: Optional[bool] = None, title: Optional[str] = None,
           do_extract: bool = True, workers: Optional[int] = None) -> IngestResult:
    doc = load(path_or_text, is_path=is_path, title=title)
    if not doc.chunks:
        return IngestResult(doc.id, doc.title, tenant, namespace, 0, 0, 0, False)

    chunk_vecs = _embed_texts(backend, [c.text for c in doc.chunks])

    if do_extract:
        extractions = _extract_all(doc.chunks, workers or config.EXTRACT_WORKERS)
    else:
        extractions = {c.id: _ex.Extraction() for c in doc.chunks}
    n_ent = sum(len(x.entities) for x in extractions.values())
    n_rel = sum(len(x.relations) for x in extractions.values())
    extracted = any(x.entities or x.relations for x in extractions.values())

    backend.load_document(doc, tenant=tenant, namespace=namespace, agent=agent,
                          chunk_vecs=chunk_vecs, extractions=extractions)
    return IngestResult(doc.id, doc.title, tenant, namespace, len(doc.chunks), n_ent, n_rel, extracted)


def recall(backend, query: str, *, tenant: str = "default", namespace: Optional[str] = None,
           k: int = 8, hops: int = 1) -> RecallResult:
    qvec = _embed_texts(backend, [query])
    chunks = backend.search(qvec, tenant=tenant, namespace=namespace, k=k)
    cids = [c["id"] for c in chunks]
    sub = backend.expand(cids, tenant=tenant, hops=hops) if cids else {"entities": [], "relations": []}
    return RecallResult(query, tenant, tuple(chunks),
                        tuple(sub.get("entities", [])), tuple(sub.get("relations", [])))
