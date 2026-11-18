"""secp256k1 curve and field constants.

All values are the canonical Bitcoin/secp256k1 parameters. They are defined
once here so the pure-Python reference, the compiled CPU backend, and the CUDA
backend can be cross-checked against a single source of truth.
"""

from __future__ import annotations

# Field prime:  p = 2^256 - 2^32 - 977
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# Group order (number of points on the curve, the "n" in  n * G = O).
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# Generator point G.
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# Number of bits in the field prime / scalar.
BITS = 256

# A well-known second point (2 * G), used as an independent self-test vector.
TWO_G_X = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
TWO_G_Y = 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A

# secp256k1 uses  a = 0,  b = 7  in  y^2 = x^3 + a x + b.
CURVE_A = 0
CURVE_B = 7

# A 64-bit mask and a helper to split a 256-bit integer into four little-endian
# 64-bit limbs (the representation used by the compiled CPU/CUDA backends).
MASK64 = (1 << 64) - 1


def to_limbs(value: int) -> tuple[int, int, int, int]:
    return (
        value & MASK64,
        (value >> 64) & MASK64,
        (value >> 128) & MASK64,
        (value >> 192) & MASK64,
    )
