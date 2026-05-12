"""Run vector-db-bench across whichever backends are installed."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_db_bench import (
    BenchmarkResult,
    ChromaBackend,
    FaissBackend,
    NumpyBruteBackend,
    benchmark,
    synthetic_dataset,
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--n-corpus", type=int, default=100_000)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=str, default="bench/results.json")
    args = p.parse_args()

    print(f"Building dataset: n_corpus={args.n_corpus}, dim={args.dim}, n_queries={args.n_queries}")
    corpus, queries = synthetic_dataset(
        n=args.n_corpus, dim=args.dim, n_queries=args.n_queries
    )

    # Gold from numpy brute.
    brute = NumpyBruteBackend()
    print("Computing gold with NumpyBruteBackend...")
    brute_result = benchmark(brute, corpus, queries, k=args.k)
    gold_ids, _ = brute.search(queries, args.k)
    print(f"Gold built in {brute_result.build_secs:.2f}s, QPS={brute_result.qps:.0f}")

    results: list[BenchmarkResult] = [brute_result]

    # FAISS-flat.
    try:
        b = FaissBackend(hnsw=False)
        results.append(benchmark(b, corpus, queries, k=args.k, gold=gold_ids))
        print(f"  faiss_flat done.")
    except Exception as exc:  # noqa: BLE001
        print(f"  faiss_flat SKIPPED: {exc}")

    # FAISS-HNSW.
    try:
        b = FaissBackend(hnsw=True)
        results.append(benchmark(b, corpus, queries, k=args.k, gold=gold_ids))
        print(f"  faiss_hnsw done.")
    except Exception as exc:  # noqa: BLE001
        print(f"  faiss_hnsw SKIPPED: {exc}")

    # Chroma.
    try:
        b = ChromaBackend()
        results.append(benchmark(b, corpus, queries, k=args.k, gold=gold_ids))
        print(f"  chroma done.")
    except Exception as exc:  # noqa: BLE001
        print(f"  chroma SKIPPED: {exc}")

    # Recall against itself for the brute baseline.
    results[0].recall_at_k = 1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}\n")

    print(f"{'backend':<15} {'build_s':>9} {'rss_MB':>8} {'QPS':>8} {'p50_us':>8} {'p99_us':>8} {'recall@k':>10}")
    for r in results:
        print(
            f"{r.backend:<15} {r.build_secs:>9.2f} {r.rss_mb_after_build:>8.1f} "
            f"{r.qps:>8.1f} {r.latency_p50_us:>8.1f} {r.latency_p99_us:>8.1f} "
            f"{r.recall_at_k if isinstance(r.recall_at_k, float) else 0:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
