"""Binary Quantization utilities for JR AutoRAG v2.

This module provides:
- Float32 to binary vector conversion (sign-threshold quantization)
- Bit packing/unpacking for efficient storage
- Dimension validation for clean byte alignment
- Versioning for reproducibility
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# Quantization version for tracking schema changes
BQ_VERSION = "1.0.0"


@dataclass
class BQConfig:
    """Configuration for binary quantization."""
    
    # Quantization rule: "sign_threshold_0" = bit=1 if value >= 0 else 0
    rule: str = "sign_threshold_0"
    
    # Whether to L2-normalize vectors before thresholding
    normalize: bool = False
    
    # Version identifier for reproducibility
    version: str = BQ_VERSION
    
    def to_dict(self) -> dict[str, str | bool]:
        return {
            "rule": self.rule,
            "normalize": self.normalize,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BQConfig":
        return cls(
            rule=data.get("rule", "sign_threshold_0"),
            normalize=data.get("normalize", False),
            version=data.get("version", BQ_VERSION),
        )


def validate_dimension(dim: int) -> bool:
    """Validate that embedding dimension is divisible by 8 for clean bit packing.
    
    Args:
        dim: Embedding dimension
        
    Returns:
        True if dimension is valid for binary quantization
    """
    return dim > 0 and dim % 8 == 0


def float32_to_binary(
    vector: np.ndarray | list[float],
    config: BQConfig | None = None,
) -> bytes:
    """Convert float32 vector to binary using sign-threshold quantization.
    
    The design explicitly expects "float32 first, then convert to binary vectors."
    
    Args:
        vector: Float32 embedding vector (1D array or list)
        config: Quantization configuration
        
    Returns:
        Packed binary bytes (1 bit per dimension)
        
    Raises:
        ValueError: If dimension is not divisible by 8
    """
    config = config or BQConfig()
    
    # Convert to numpy if needed
    if isinstance(vector, list):
        arr = np.array(vector, dtype=np.float32)
    else:
        arr = vector.astype(np.float32)
    
    dim = len(arr)
    if not validate_dimension(dim):
        raise ValueError(
            f"Embedding dimension {dim} must be divisible by 8 for binary quantization. "
            f"Consider padding or using a different embedding model."
        )
    
    # Optional L2 normalization before thresholding
    if config.normalize:
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
    
    # Apply quantization rule
    if config.rule == "sign_threshold_0":
        # Baseline: bit = 1 if value >= 0 else 0
        bits = (arr >= 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown quantization rule: {config.rule}")
    
    # Pack bits into bytes (8 bits per byte)
    return _pack_bits(bits)


def binary_to_bits(binary_vector: bytes, dim: int) -> np.ndarray:
    """Unpack binary bytes back to bit array.
    
    Args:
        binary_vector: Packed binary bytes
        dim: Original embedding dimension
        
    Returns:
        Numpy array of 0s and 1s
    """
    return _unpack_bits(binary_vector, dim)


def _pack_bits(bits: np.ndarray) -> bytes:
    """Pack bit array into bytes (MSB first within each byte).
    
    Args:
        bits: 1D numpy array of 0s and 1s
        
    Returns:
        Packed bytes
    """
    # Reshape to groups of 8 bits
    bits_reshaped = bits.reshape(-1, 8)
    
    # Pack each group of 8 bits into a byte (MSB first)
    # bit positions: [7, 6, 5, 4, 3, 2, 1, 0] for each byte
    powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    packed = (bits_reshaped * powers).sum(axis=1).astype(np.uint8)
    
    return packed.tobytes()


def _unpack_bits(packed: bytes, dim: int) -> np.ndarray:
    """Unpack bytes back to bit array.
    
    Args:
        packed: Packed binary bytes
        dim: Target dimension (number of bits)
        
    Returns:
        1D numpy array of 0s and 1s
    """
    arr = np.frombuffer(packed, dtype=np.uint8)
    
    # Unpack each byte to 8 bits (MSB first)
    bits = np.unpackbits(arr, bitorder='big')
    
    return bits[:dim]


def hamming_distance(a: bytes, b: bytes) -> int:
    """Compute Hamming distance between two binary vectors.
    
    Args:
        a: First binary vector (packed bytes)
        b: Second binary vector (packed bytes)
        
    Returns:
        Number of differing bits
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    
    arr_a = np.frombuffer(a, dtype=np.uint8)
    arr_b = np.frombuffer(b, dtype=np.uint8)
    
    # XOR and count set bits
    xor = np.bitwise_xor(arr_a, arr_b)
    return int(np.unpackbits(xor).sum())


def batch_float32_to_binary(
    vectors: np.ndarray | list[list[float]],
    config: BQConfig | None = None,
) -> list[bytes]:
    """Convert batch of float32 vectors to binary.
    
    Args:
        vectors: 2D array of shape (n_vectors, dim) or list of lists
        config: Quantization configuration
        
    Returns:
        List of packed binary bytes
    """
    config = config or BQConfig()
    
    if isinstance(vectors, list):
        arr = np.array(vectors, dtype=np.float32)
    else:
        arr = vectors.astype(np.float32)
    
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    dim = arr.shape[1]
    if not validate_dimension(dim):
        raise ValueError(
            f"Embedding dimension {dim} must be divisible by 8 for binary quantization."
        )
    
    # Optional L2 normalization
    if config.normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        arr = arr / norms
    
    # Apply quantization rule
    if config.rule == "sign_threshold_0":
        bits = (arr >= 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown quantization rule: {config.rule}")
    
    # Pack each row
    return [_pack_bits(row) for row in bits]


def get_binary_dimension(float_dim: int) -> int:
    """Get the number of bytes needed to store a binary vector.
    
    Args:
        float_dim: Original float32 embedding dimension
        
    Returns:
        Number of bytes for binary representation
    """
    if not validate_dimension(float_dim):
        raise ValueError(f"Dimension {float_dim} must be divisible by 8")
    return float_dim // 8


def estimate_storage_savings(
    num_vectors: int,
    embedding_dim: int,
) -> dict[str, int | float]:
    """Estimate storage savings from binary quantization.
    
    Args:
        num_vectors: Number of vectors to store
        embedding_dim: Embedding dimension
        
    Returns:
        Dict with storage estimates in bytes and savings ratio
    """
    float32_bytes = num_vectors * embedding_dim * 4  # 4 bytes per float32
    binary_bytes = num_vectors * (embedding_dim // 8)  # 1 bit per dimension
    
    return {
        "float32_bytes": float32_bytes,
        "binary_bytes": binary_bytes,
        "savings_ratio": float32_bytes / binary_bytes if binary_bytes > 0 else 0,
        "savings_percent": (1 - binary_bytes / float32_bytes) * 100 if float32_bytes > 0 else 0,
    }


__all__ = [
    "BQConfig",
    "BQ_VERSION",
    "validate_dimension",
    "float32_to_binary",
    "binary_to_bits",
    "hamming_distance",
    "batch_float32_to_binary",
    "get_binary_dimension",
    "estimate_storage_savings",
]
