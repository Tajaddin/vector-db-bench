# vector-db-bench

> Reproducible head-to-head benchmark of vector-DB backends on **QPS, recall@10, p99 latency, and RSS memory**. Pure Python harness, no Docker required. **FAISS-HNSW: 15× QPS lift over brute-force at 93% recall@10**, on 50K × 256-D synthetic vectors.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-5%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

50,000 vectors × 256 dimensions, 200 queries, k=10, single CPU core, in-process for every backend.

| Backend | Build (s) | RSS (MB) | QPS | p50 (µs) | p99 (µs) | recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| `numpy_brute` (gold) | 0.03 | 142.7 | 365 | 2,686 | 3,805 | **1.000** |
| `faiss_flat` (exact IP) | 0.08 | 215.2 | 237 | 4,116 | 6,165 | 1.000 |
| **`faiss_hnsw`** | 1.80 | 226.7 | **3,583** | **258** | 485 | 0.929 |
| `chroma` (HNSW + cosine) | 8.14 | 329.2 | 112 | 8,607 | 15,648 | 0.982 |

**Read this as:**

* `numpy_brute` is the gold baseline — exact recall, no library required, but 10× slower than FAISS HNSW.
* `faiss_flat` matches gold recall but is slower on this hardware due to extra memory layout work.
* **`faiss_hnsw` is the right default for medium-scale RAG**: 15× QPS over brute at 92.9% recall@10. Build cost rises to 1.8s but is a one-shot fixed cost.
* `chroma` ships an HNSW index too but with full ID/metadata bookkeeping — slower queries in exchange for production amenities (persistence, metadata filters, server mode).

Raw output: [`bench/results.json`](bench/results.json). Reproduce: `python bench/run_bench.py`.

## Why this exists

Picking a vector DB for a production RAG system is a real platform decision and the vendor benchmarks are all flavored. This project is the smallest possible neutral comparison harness:

* **Same dataset for every backend** — synthetic 50-cluster mixture on the unit sphere, so HNSW has non-trivial structure to work with.
* **Same gold** — gold neighbors computed once via `NumpyBruteBackend`. Every other backend's recall is `|retrieved ∩ gold| / k`.
* **Same hardware** — pure Python harness, no Docker, no remote servers. The numbers above came from a single Python process; what you measure is what your code will see.

It's not the deepest benchmark out there — it's the one that runs in 30 seconds with `pip install` and shows you the actual trade-off.

## Quickstart

```bash
pip install -e ".[all]"            # numpy_brute + FAISS + Chroma
python bench/run_bench.py
```

```python
from vector_db_bench import (
    FaissBackend, ChromaBackend, NumpyBruteBackend,
    synthetic_dataset, benchmark,
)

corpus, queries = synthetic_dataset(n=50_000, dim=256, n_queries=200)

# Gold (also acts as baseline result)
brute = NumpyBruteBackend()
gold_result = benchmark(brute, corpus, queries, k=10)
gold_ids, _ = brute.search(queries, k=10)

# Compare any backend against gold
faiss = FaissBackend(hnsw=True)
result = benchmark(faiss, corpus, queries, k=10, gold=gold_ids)
print(result.qps, result.recall_at_k)
```

## Backends

| Backend | Index type | Notes |
|---|---|---|
| `NumpyBruteBackend` | normalized dot product | exact, used to generate gold |
| `FaissBackend(hnsw=False)` | `IndexFlatIP` | exact, FAISS reference |
| `FaissBackend(hnsw=True, hnsw_M=16, hnsw_ef=64)` | `IndexHNSWFlat` | the production-ready ANN default |
| `ChromaBackend` | HNSW + cosine | uses `chromadb.EphemeralClient` for an in-memory comparison |

All four implement the same surface:

```python
backend.add(vectors: np.ndarray, ids: list[str])
backend.search(queries: np.ndarray, k: int) -> (idx, scores)
```

`idx` are integer offsets into the ID list, so recall is computed without string round-trips.

## Tuning HNSW

```python
b = FaissBackend(hnsw=True, hnsw_M=32, hnsw_ef=128)
# Higher M = more memory, better recall.
# Higher efSearch (ef) = slower queries, better recall.
```

Defaults (`M=16`, `ef=64`) target the "good enough for retrieval" sweet spot. Production deployments often run `M=32 / ef=128-256` and accept the memory + latency cost for recall ≥ 0.97.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
5 passed
```

Covers:
* Synthetic dataset returns float32 L2-normalized vectors of the right shape
* NumpyBrute returns the right top-k with scores in non-increasing order
* Recall is 1.0 when comparing brute against itself
* Recall drops to ~k/n when comparing random vs gold
* `benchmark()` returns a populated `BenchmarkResult`

## Project layout

```
.
├── src/vector_db_bench/
│   ├── backends.py     # NumpyBrute + FAISS (flat & HNSW) + Chroma
│   ├── dataset.py      # synthetic mixture-of-clusters + recall@k math
│   └── bench.py        # benchmark() harness: build + per-query + recall
├── tests/              # 5 pytest cases
└── bench/
    ├── run_bench.py
    └── results.json    # latest run
```

## Limitations

**Synthetic data.** Real-world embedding distributions have heavier tails and tighter clusters than the synthetic mixture. The relative ranking of backends usually holds on real data; absolute QPS and recall numbers will shift.

**No Qdrant or pgvector.** Both require running a server (Qdrant local / Postgres + pgvector). The harness has an extras hook (`pip install vector-db-bench[qdrant]`) and a stub backend slot, but a polished implementation would need docker-compose — out of scope for v0.1.

**No filtering.** Production RAG often filters by metadata at query time (e.g., `user_id == X`). Chroma supports this natively; FAISS needs the IDs-with-selector pattern. Not benchmarked here.

**Single-thread, single-process.** Concurrent queries (e.g., 10 simultaneous searches) would change the relative numbers — FAISS releases the GIL during search, Chroma's Python overhead means it can be the bottleneck.

**Disk persistence ignored.** All backends here run in-memory. Chroma's persistent client and FAISS's `write_index` aren't measured. For a load-time / cold-start benchmark, extend `bench.py`.

## License

MIT — see [LICENSE](LICENSE).
