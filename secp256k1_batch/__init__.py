"""Batched secp256k1 public-key generation across CPU and GPU.

The primary entry point is :func:`generate_public_keys`, which produces the
public keys for a contiguous range of private keys using either a compiled C
CPU backend, a CUDA GPU backend, or a pure-Python reference implementation.
"""

from .constants import GX, GY, N, P
from .engine import generate_public_keys, iter_keys, iter_public_key_chunks

__all__ = [
    "generate_public_keys",
    "iter_keys",
    "iter_public_key_chunks",
    "P",
    "N",
    "GX",
    "GY",
]
