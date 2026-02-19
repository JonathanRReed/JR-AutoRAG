"""Tests for Binary Quantization module.

Tests cover:
- Float32 to binary conversion
- Bit packing/unpacking
- Hamming distance calculation
- Dimension validation
- Storage savings estimation
"""

import numpy as np
import pytest

from app.core.binary_quantization import (
    BQ_VERSION,
    BQConfig,
    batch_float32_to_binary,
    binary_to_bits,
    estimate_storage_savings,
    float32_to_binary,
    get_binary_dimension,
    hamming_distance,
    validate_dimension,
)


class TestBQConfig:
    def test_default_config(self):
        config = BQConfig()
        assert config.rule == "sign_threshold_0"
        assert config.normalize is False
        assert config.version == BQ_VERSION

    def test_to_dict(self):
        config = BQConfig(normalize=True)
        d = config.to_dict()
        assert d["rule"] == "sign_threshold_0"
        assert d["normalize"] is True
        assert d["version"] == BQ_VERSION

    def test_from_dict(self):
        data = {"rule": "sign_threshold_0", "normalize": True, "version": "1.0.0"}
        config = BQConfig.from_dict(data)
        assert config.normalize is True


class TestValidateDimension:
    def test_valid_dimensions(self):
        assert validate_dimension(8) is True
        assert validate_dimension(768) is True
        assert validate_dimension(1024) is True
        assert validate_dimension(384) is True

    def test_invalid_dimensions(self):
        assert validate_dimension(0) is False
        assert validate_dimension(7) is False
        assert validate_dimension(769) is False
        assert validate_dimension(-8) is False


class TestFloat32ToBinary:
    def test_basic_conversion(self):
        # Simple case: 8 values, half positive, half negative
        vector = [1.0, -1.0, 0.5, -0.5, 0.0, -0.1, 0.1, -0.01]
        result = float32_to_binary(vector)

        # Expected: [1, 0, 1, 0, 1, 0, 1, 0] -> 0b10101010 = 170
        assert len(result) == 1
        assert result[0] == 170

    def test_all_positive(self):
        vector = [1.0] * 8
        result = float32_to_binary(vector)
        # All 1s -> 0b11111111 = 255
        assert result[0] == 255

    def test_all_negative(self):
        vector = [-1.0] * 8
        result = float32_to_binary(vector)
        # All 0s -> 0b00000000 = 0
        assert result[0] == 0

    def test_zeros_are_positive(self):
        # Zero should be treated as >= 0, so bit = 1
        vector = [0.0] * 8
        result = float32_to_binary(vector)
        assert result[0] == 255

    def test_768_dim_vector(self):
        # Typical embedding dimension
        np.random.seed(42)
        vector = np.random.randn(768).tolist()
        result = float32_to_binary(vector)

        assert len(result) == 768 // 8
        assert len(result) == 96

    def test_numpy_input(self):
        vector = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        result = float32_to_binary(vector)
        assert len(result) == 1

    def test_invalid_dimension_raises(self):
        vector = [1.0] * 7  # Not divisible by 8
        with pytest.raises(ValueError, match="divisible by 8"):
            float32_to_binary(vector)

    def test_with_normalization(self):
        config = BQConfig(normalize=True)
        vector = [10.0, -10.0, 5.0, -5.0, 0.0, -1.0, 1.0, -0.5]
        result = float32_to_binary(vector, config)

        # After normalization, signs should be preserved
        assert len(result) == 1


class TestBinaryToBits:
    def test_roundtrip(self):
        original = [1.0, -1.0, 0.5, -0.5, 0.0, -0.1, 0.1, -0.01]
        binary = float32_to_binary(original)
        bits = binary_to_bits(binary, 8)

        expected = [1, 0, 1, 0, 1, 0, 1, 0]
        assert list(bits) == expected

    def test_768_dim_roundtrip(self):
        np.random.seed(42)
        original = np.random.randn(768)
        binary = float32_to_binary(original)
        bits = binary_to_bits(binary, 768)

        # Verify bits match sign of original
        expected_bits = (original >= 0).astype(np.uint8)
        np.testing.assert_array_equal(bits, expected_bits)


class TestHammingDistance:
    def test_identical_vectors(self):
        a = float32_to_binary([1.0] * 8)
        b = float32_to_binary([1.0] * 8)
        assert hamming_distance(a, b) == 0

    def test_opposite_vectors(self):
        a = float32_to_binary([1.0] * 8)
        b = float32_to_binary([-1.0] * 8)
        assert hamming_distance(a, b) == 8

    def test_half_different(self):
        a = float32_to_binary([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
        b = float32_to_binary([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        assert hamming_distance(a, b) == 8

    def test_one_bit_different(self):
        a = float32_to_binary([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        b = float32_to_binary([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
        assert hamming_distance(a, b) == 1

    def test_length_mismatch_raises(self):
        a = float32_to_binary([1.0] * 8)
        b = float32_to_binary([1.0] * 16)
        with pytest.raises(ValueError, match="length mismatch"):
            hamming_distance(a, b)


class TestBatchFloat32ToBinary:
    def test_batch_conversion(self):
        vectors = [
            [1.0] * 8,
            [-1.0] * 8,
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ]
        results = batch_float32_to_binary(vectors)

        assert len(results) == 3
        assert results[0][0] == 255  # All positive
        assert results[1][0] == 0    # All negative
        assert results[2][0] == 170  # Alternating

    def test_numpy_batch(self):
        np.random.seed(42)
        vectors = np.random.randn(100, 768)
        results = batch_float32_to_binary(vectors)

        assert len(results) == 100
        assert all(len(r) == 96 for r in results)


class TestGetBinaryDimension:
    def test_common_dimensions(self):
        assert get_binary_dimension(768) == 96
        assert get_binary_dimension(384) == 48
        assert get_binary_dimension(1024) == 128
        assert get_binary_dimension(8) == 1

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            get_binary_dimension(7)


class TestEstimateStorageSavings:
    def test_storage_estimate(self):
        result = estimate_storage_savings(1000, 768)

        # Float32: 1000 * 768 * 4 = 3,072,000 bytes
        assert result["float32_bytes"] == 3_072_000

        # Binary: 1000 * 96 = 96,000 bytes
        assert result["binary_bytes"] == 96_000

        # Savings ratio: 32x
        assert result["savings_ratio"] == 32.0

        # Savings percent: ~96.875%
        assert result["savings_percent"] == pytest.approx(96.875)

    def test_large_scale(self):
        # PubMed-scale: ~36M items
        result = estimate_storage_savings(36_000_000, 768)

        # Float32: ~110 GB
        assert result["float32_bytes"] == 36_000_000 * 768 * 4

        # Binary: ~3.4 GB
        assert result["binary_bytes"] == 36_000_000 * 96

        assert result["savings_ratio"] == 32.0
