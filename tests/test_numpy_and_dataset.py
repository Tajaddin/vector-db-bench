"""Tests for the numpy brute backend, dataset, and recall math."""

from __future__ import annotations

import numpy as np
import pytest

from vector_db_bench import NumpyBruteBackend, benchmark, gold_recall, synthetic_dataset


def test_synthetic_dataset_shapes_and_normalization() -> None:
    corpus, queries = synthetic_dataset(n=500, dim=64, n_clusters=10, n_queries=20, seed=1)
    assert corpus.shape == (500, 64)
    assert queries.shape == (20, 64)
    assert corpus.dtype == np.float32
    assert queries.dtype == np.float32
    norms = np.linalg.norm(corpus, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_numpy_brute_returns_k_neighbors() -> None:
    corpus, queries = synthetic_dataset(n=200, dim=32, n_clusters=5, n_queries=10, seed=2)
    brute = NumpyBruteBackend()
    brute.add(corpus, [str(i) for i in range(len(corpus))])
    idx, scores = brute.search(queries, k=5)
    assert idx.shape == (10, 5)
    assert scores.shape == (10, 5)
    # Scores must be non-increasing within each row.
    assert (scores[:, :-1] >= scores[:, 1:] - 1e-5).all()


def test_recall_perfect_against_itself() -> None:
    corpus, queries = synthetic_dataset(n=200, dim=32, n_queries=10, seed=3)
    brute = NumpyBruteBackend()
    brute.add(corpus, [str(i) for i in range(len(corpus))])
    idx, _ = brute.search(queries, k=5)
    # Comparing brute against brute → perfect recall.
    assert gold_recall(idx, idx, k=5) == pytest.approx(1.0)


def test_recall_drops_with_random_neighbors() -> None:
    corpus, queries = synthetic_dataset(n=200, dim=32, n_queries=10, seed=4)
    brute = NumpyBruteBackend()
    brute.add(corpus, [str(i) for i in range(len(corpus))])
    idx, _ = brute.search(queries, k=5)
    random_idx = np.random.default_rng(0).integers(0, 200, size=(10, 5))
    recall = gold_recall(random_idx, idx, k=5)
    assert recall < 0.5  # random retrieval is ~k/n = 2.5%


def test_benchmark_returns_populated_result() -> None:
    corpus, queries = synthetic_dataset(n=200, dim=32, n_queries=10, seed=5)
    brute = NumpyBruteBackend()
    res = benchmark(brute, corpus, queries, k=5)
    assert res.backend == "numpy_brute"
    assert res.qps > 0
    assert res.build_secs >= 0
    assert res.latency_p50_us > 0
