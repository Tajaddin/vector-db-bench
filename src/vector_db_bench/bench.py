"""Benchmark harness.

For one backend, run:
* index build (timed + RSS measured)
* per-query latency over the query set
* recall@k vs the gold neighbors

Latency is reported as p50/p95/p99/max in microseconds; QPS is total
queries divided by total wall time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import psutil

from vector_db_bench.backends import VectorBackend
from vector_db_bench.dataset import gold_recall


@dataclass
class BenchmarkResult:
    backend: str
    n_corpus: int
    n_queries: int
    k: int
    build_secs: float
    rss_mb_after_build: float
    qps: float
    latency_p50_us: float
    latency_p95_us: float
    latency_p99_us: float
    latency_max_us: float
    recall_at_k: float


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1_000_000


def _quantile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[int(q * (len(s) - 1))]


def benchmark(
    backend: VectorBackend,
    corpus: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 10,
    gold: np.ndarray | None = None,
    name_override: str | None = None,
) -> BenchmarkResult:
    """Run a full benchmark cycle for one backend."""
    # 1. Index build.
    rss_before = _rss_mb()
    ids = [str(i) for i in range(len(corpus))]
    t0 = time.perf_counter()
    backend.add(corpus, ids)
    build_secs = time.perf_counter() - t0
    rss_after = _rss_mb()

    # 2. Per-query latency.
    latencies_us: list[float] = []
    t0 = time.perf_counter()
    for q in queries:
        q2 = q.reshape(1, -1)
        ts = time.perf_counter_ns()
        backend.search(q2, k)
        latencies_us.append((time.perf_counter_ns() - ts) / 1000)
    total_secs = time.perf_counter() - t0

    # 3. Bulk recall@k vs gold.
    if gold is not None:
        retrieved, _ = backend.search(queries, k)
        recall = gold_recall(retrieved, gold, k)
    else:
        recall = float("nan")

    return BenchmarkResult(
        backend=name_override or backend.name,
        n_corpus=len(corpus),
        n_queries=len(queries),
        k=k,
        build_secs=round(build_secs, 3),
        rss_mb_after_build=round(rss_after, 1),
        qps=round(len(queries) / max(total_secs, 1e-9), 1),
        latency_p50_us=round(_quantile(latencies_us, 0.50), 2),
        latency_p95_us=round(_quantile(latencies_us, 0.95), 2),
        latency_p99_us=round(_quantile(latencies_us, 0.99), 2),
        latency_max_us=round(max(latencies_us), 2),
        recall_at_k=round(recall, 4) if recall == recall else float("nan"),
    )
