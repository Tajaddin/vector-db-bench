"""Reproducible head-to-head benchmark of vector DB backends."""

from vector_db_bench.backends import (
    BackendStats,
    ChromaBackend,
    FaissBackend,
    NumpyBruteBackend,
    VectorBackend,
)
from vector_db_bench.bench import BenchmarkResult, benchmark
from vector_db_bench.dataset import gold_recall, synthetic_dataset

__version__ = "0.1.0"

__all__ = [
    "VectorBackend",
    "BackendStats",
    "NumpyBruteBackend",
    "FaissBackend",
    "ChromaBackend",
    "benchmark",
    "BenchmarkResult",
    "synthetic_dataset",
    "gold_recall",
]
