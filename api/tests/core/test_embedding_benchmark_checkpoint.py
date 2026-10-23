"""Benchmark restart state must never mix models, fixtures, or corrupt vectors."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("embedding_benchmark", Path(__file__).parents[3] / "scripts/embedding-quality-benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_checkpoint_round_trip_and_identity(tmp_path):
    path = tmp_path / "state.npz"
    vectors = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    benchmark.save_checkpoint(path, "profile:fixture", vectors, 12.5)
    restored, elapsed = benchmark.load_checkpoint(path, "profile:fixture", 3)
    np.testing.assert_array_equal(restored, vectors)
    assert elapsed == 12.5
    with pytest.raises(ValueError, match="identity"):
        benchmark.load_checkpoint(path, "different", 3)


def test_checkpoint_rejects_corrupt_or_excess_vectors(tmp_path):
    path = tmp_path / "state.npz"
    benchmark.save_checkpoint(path, "same", np.ones((4, 2)), 1.)
    with pytest.raises(ValueError, match="vectors"):
        benchmark.load_checkpoint(path, "same", 3)
    benchmark.save_checkpoint(path, "same", np.array([[np.nan]]), 1.)
    with pytest.raises(ValueError, match="vectors"):
        benchmark.load_checkpoint(path, "same", 3)
