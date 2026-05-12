"""Synthetic vector dataset + gold-recall computation.

Vectors are sampled from a mixture of K Gaussian clusters on the unit sphere
so the ANN backends see a non-trivial, multi-modal distribution. The gold
neighbors are computed once via the :class:`NumpyBruteBackend` and reused
across backends to score recall@k.
"""

from __future__ import annotations

import numpy as np


def synthetic_dataset(
    *,
    n: int = 100_000,
    dim: int = 256,
    n_clusters: int = 50,
    n_queries: int = 200,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(corpus_vectors, query_vectors)``.

    Both arrays are float32 and L2-normalized so every backend sees the same
    cosine geometry.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    cluster_ids = rng.integers(0, n_clusters, size=n)
    noise = rng.normal(scale=0.1, size=(n, dim)).astype(np.float32)
    corpus = centers[cluster_ids] + noise
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12

    # Queries: pick random corpus points and add small noise — emulates "user
    # types a phrase similar to a stored doc."
    query_idx = rng.choice(n, n_queries, replace=False)
    query_noise = rng.normal(scale=0.05, size=(n_queries, dim)).astype(np.float32)
    queries = corpus[query_idx] + query_noise
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    return corpus, queries


def gold_recall(retrieved: np.ndarray, gold: np.ndarray, k: int) -> float:
    """Mean recall@k against gold top-k. Both arrays shape ``(n_queries, k)``."""
    n = retrieved.shape[0]
    total = 0.0
    for i in range(n):
        set_r = set(retrieved[i].tolist())
        set_g = set(gold[i].tolist())
        total += len(set_r & set_g) / max(len(set_g), 1)
    return total / max(n, 1)
