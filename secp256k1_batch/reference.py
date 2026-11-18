"""Pure-Python secp256k1 reference implementation.

This module is deliberately simple and obviously correct: it uses Python's
arbitrary-precision integers and the textbook affine-coordinate formulas. It is
the correctness oracle that the compiled CPU backend and the CUDA backend are
validated against, and it doubles as a dependency-free CPU fallback.

It is *not* fast. Use ``cpu`` or ``gpu`` backends for production throughput.
"""

from __future__ import annotations

from .constants import CURVE_A, CURVE_B, GX, GY, N, P

# A point is represented as ``(x, y)`` or ``None`` for the point at infinity.
Point = tuple[int, int] | None


def _mod(a: int, m: int = P) -> int:
    return a % m


def field_inv(x: int) -> int:
    """Modular inverse in F_p via Fermat's little theorem (x^(p-2) mod p)."""
    if x % P == 0:
        raise ZeroDivisionError("0 has no inverse in F_p")
    return pow(x, P - 2, P)


def field_add(a: int, b: int) -> int:
    return (a + b) % P


def field_sub(a: int, b: int) -> int:
    return (a - b) % P


def field_mul(a: int, b: int) -> int:
    return (a * b) % P


def field_sqr(a: int) -> int:
    return (a * a) % P


def point_neg(point: Point) -> Point:
    if point is None:
        return None
    x, y = point
    return (x, (-y) % P)


def point_add(left: Point, right: Point) -> Point:
    """Affine point addition (handles infinity and doubling)."""
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None  # left == -right  ->  point at infinity
        # Doubling.
        lam = (3 * x1 * x1 + CURVE_A) * field_inv(2 * y1) % P
    else:
        lam = (y2 - y1) * field_inv(x2 - x1) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def point_double(point: Point) -> Point:
    return point_add(point, point)


def scalar_mult(k: int, point: Point = (GX, GY)) -> Point:
    """Double-and-add scalar multiplication  result = k * point."""
    k %= N
    result: Point = None
    addend: Point = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_double(addend)
        k >>= 1
    return result


def public_key(private_key: int) -> Point:
    """Derive the uncompressed public point for a private key  k * G."""
    if not 0 < private_key < N:
        raise ValueError("private key must be in [1, n-1]")
    return scalar_mult(private_key)


def public_keys_range(start: int, count: int) -> list[Point]:
    """Return the public points for  start, start+1, ..., start+count-1.

    Uses the increment-by-generator trick: only the first point needs a full
    scalar multiplication; every following point is a single point addition.
    """
    if not 0 < start < N:
        raise ValueError("start must be in [1, n-1]")
    if count < 0:
        raise ValueError("count must be non-negative")
    out: list[Point] = []
    current = scalar_mult(start)
    out.append(current)
    for _ in range(count - 1):
        current = point_add(current, (GX, GY))
        out.append(current)
    return out


def public_keys_range_grouped(start: int, count: int, grp_size: int = 256) -> list[Point]:
    """Grouped-affine reference that mirrors the CUDA kernel's algorithm.

    Each block of ``grp_size`` keys is generated around an affine centre
    ``S = (start + blk + grp_size/2) * G``, batch-inverting the deltas
    ``(d*G).x - S.x`` and reusing each inverse for ``S - d*G`` and ``S + d*G``.
    The centre marches forward by ``grp_size * G`` per block (with the
    ``2*stride`` special case when the centre equals the stride point).  This
    is a pure-Python cross-check of the GPU kernel's maths (it is *not* the
    fast path; it still uses one inversion per delta via Fermat).
    """
    if grp_size < 2 or grp_size % 2:
        raise ValueError("grp_size must be an even integer >= 2")
    hsize = grp_size // 2

    # Generator table: (i+1)*G for i = 0 .. hsize-1  (affine).
    table: list[Point] = []
    cur: Point = (GX, GY)
    for _ in range(hsize):
        table.append(cur)
        cur = point_add(cur, (GX, GY))
    # stride = grp_size * G = 2 * (hsize * G) ; stride2 = 2 * stride.
    stride = point_double(table[hsize - 1])
    stride2 = point_double(stride)

    out: list[Point] = []
    blocks = (count + grp_size - 1) // grp_size

    sx, sy = scalar_mult((start + hsize) % N)

    for b in range(blocks):
        blk = b * grp_size

        invs: list[int] = [field_inv((gx_i - sx) % P) for gx_i, _gy_i in table]

        keys: list[Point | None] = [None] * grp_size
        keys[hsize] = (sx, sy)
        for d in range(1, hsize + 1):
            i = d - 1
            gx_i, gy_i = table[i]
            inv = invs[i]

            # negative side: S - d*G
            lam = ((-gy_i - sy) % P) * inv % P
            rx = (lam * lam - sx - gx_i) % P
            ry = (lam * (sx - rx) - sy) % P
            keys[hsize - d] = (rx, ry)

            # positive side: S + d*G  (d <= hsize - 1)
            if d <= hsize - 1:
                lam = ((gy_i - sy) % P) * inv % P
                rx = (lam * lam - sx - gx_i) % P
                ry = (lam * (sx - rx) - sy) % P
                keys[hsize + d] = (rx, ry)

        out.extend(keys[: min(grp_size, count - blk)])

        # advance centre: S += stride (or 2*stride if S == stride)
        if (stride[0] - sx) % P == 0:
            sx, sy = stride2
        else:
            inv = field_inv((stride[0] - sx) % P)
            lam = ((stride[1] - sy) % P) * inv % P
            old_sx = sx
            sx_new = (lam * lam - old_sx - stride[0]) % P
            sy = (lam * (old_sx - sx_new) - sy) % P
            sx = sx_new

    return out
