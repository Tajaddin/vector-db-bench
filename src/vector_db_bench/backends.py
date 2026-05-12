"""Three vector-DB backends behind one tiny interface.

* :class:`NumpyBruteBackend`  — pure numpy brute-force search; the recall@k
  baseline (always 1.0). Used to compute gold neighbors.
* :class:`FaissBackend`        — IndexFlatIP (exact) or IndexHNSWFlat (ANN).
* :class:`ChromaBackend`       — in-memory persistent client; uses the
  default cosine HNSW index Chroma ships with.

Every backend implements ``add(vectors, ids)`` and ``search(queries, k)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@dataclass
class BackendStats:
    name: str
    build_secs: float
    rss_mb_at_build_end: float


@runtime_checkable
class VectorBackend(Protocol):
    name: str

    def add(self, vectors: np.ndarray, ids: list[str]) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(neighbor_ids_int_indices, scores)`` of shape ``(n_queries, k)``.

        ``neighbor_ids_int_indices`` are integer offsets into the ID list used
        when adding. Callers convert to string IDs via the offset list they
        own. We don't return strings here because backends differ in storage.
        """
        ...


# ---------------------------------------------------------------------------
# NumpyBruteBackend
# ---------------------------------------------------------------------------


class NumpyBruteBackend:
    """Pure-numpy brute-force cosine-similarity search.

    The reference recall baseline. Use this to produce gold neighbors and to
    sanity-check ANN backends.
    """

    name = "numpy_brute"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        # Normalize once at insert time so search is just a dot product.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vectors / norms
        self._ids = ids

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._vectors is not None, "call add() first"
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        qn = np.linalg.norm(queries, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        queries = queries / qn
        # Cosine similarity → top-k argpartition.
        scores = queries @ self._vectors.T
        top_part = np.argpartition(-scores, k, axis=1)[:, :k]
        # Sort the top-k slice.
        ranked = np.empty_like(top_part)
        ranked_scores = np.empty_like(scores[:, :k])
        for i in range(queries.shape[0]):
            idx = top_part[i]
            order = np.argsort(-scores[i, idx])
            ranked[i] = idx[order]
            ranked_scores[i] = scores[i, ranked[i]]
        return ranked, ranked_scores


# ---------------------------------------------------------------------------
# FaissBackend
# ---------------------------------------------------------------------------


class FaissBackend:
    """FAISS IndexFlatIP (exact) by default; switch to HNSWFlat via ``hnsw=True``."""

    def __init__(self, *, hnsw: bool = False, hnsw_M: int = 16, hnsw_ef: int = 64) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss-cpu is not installed. `pip install vector-db-bench[faiss]`."
            ) from exc
        self.hnsw = hnsw
        self.hnsw_M = hnsw_M
        self.hnsw_ef = hnsw_ef
        self.name = f"faiss_{'hnsw' if hnsw else 'flat'}"
        self._index = None
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        import faiss

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = (vectors / norms).copy()
        dim = vectors.shape[1]
        if self.hnsw:
            index = faiss.IndexHNSWFlat(dim, self.hnsw_M)
            index.hnsw.efConstruction = self.hnsw_ef
            index.hnsw.efSearch = self.hnsw_ef
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._ids = ids

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None, "call add() first"
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        qn = np.linalg.norm(queries, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        queries = (queries / qn).copy()
        scores, idx = self._index.search(queries, k)
        return idx, scores


# ---------------------------------------------------------------------------
# ChromaBackend
# ---------------------------------------------------------------------------


class ChromaBackend:
    """In-memory chromadb backend. Uses cosine distance by default."""

    name = "chroma"

    def __init__(self) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "chromadb is not installed. `pip install vector-db-bench[chroma]`."
            ) from exc
        self._collection = None
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str], batch_size: int = 5000) -> None:
        import chromadb

        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("bench")
        except Exception:  # noqa: BLE001
            pass
        coll = client.create_collection(
            name="bench", metadata={"hnsw:space": "cosine"}
        )
        # Chroma has a per-batch limit on add(); chunk so big corpora work.
        v_list = vectors.astype(np.float32).tolist()
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            coll.add(ids=ids[start:end], embeddings=v_list[start:end])
        self._collection = coll
        self._ids = ids

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._collection is not None, "call add() first"
        out_ids = np.zeros((len(queries), k), dtype=np.int64)
        out_scores = np.zeros((len(queries), k), dtype=np.float32)
        # Map string IDs back to integer index in self._ids.
        id_to_idx = {sid: i for i, sid in enumerate(self._ids)}
        for q_i, q in enumerate(queries.astype(np.float32)):
            res = self._collection.query(query_embeddings=[q.tolist()], n_results=k)
            for r_i, sid in enumerate(res["ids"][0]):
                out_ids[q_i, r_i] = id_to_idx.get(sid, -1)
                # Chroma returns distance; turn into similarity for parity.
                dist = res["distances"][0][r_i] if "distances" in res else 0.0
                out_scores[q_i, r_i] = 1.0 - float(dist)
        return out_ids, out_scores
