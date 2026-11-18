"""Unified entry point for batched secp256k1 public-key generation.

Selects a backend (GPU, compiled CPU, or the pure-Python reference) and returns
the public keys for a contiguous private-key range as raw bytes:

  * uncompressed: count * 64 bytes, each key is x || y (32-byte big-endian).
  * compressed:   count * 33 bytes, each key is 0x02/0x03 || x.
"""

from __future__ import annotations

from .constants import N

BACKENDS = ("auto", "gpu", "cpu", "reference")


def _validate(start: int, count: int) -> None:
    if not isinstance(start, int) or not isinstance(count, int):
        raise TypeError("start and count must be integers")
    if not 0 < start < N:
        raise ValueError("start must be in [1, n-1]")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count and start + count - 1 > 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF:
        raise ValueError("range exceeds 256 bits")


_BACKEND_CACHE: dict = {}


def _make_backend(name: str, opts: dict):
    """Build (and cache) a backend instance.

    The GPU backend compiles the CUDA kernel with NVRTC at construction, which
    can take several seconds. Caching the instance means that cost is paid once
    per process instead of once per ``generate_public_keys`` call.
    """
    key = (name, tuple(sorted(opts.items())))
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    backend: object
    if name == "reference":
        from .reference import public_keys_range

        class _Ref:
            name = "reference"

            def generate(self, start, count, compressed=False, nthreads=0):
                pts = public_keys_range(start, count)
                if compressed:
                    out = bytearray(count * 33)
                    for i, (x, y) in enumerate(pts):
                        out[i * 33] = 0x02 | (y & 1)
                        out[i * 33 + 1 : (i + 1) * 33] = x.to_bytes(32, "big")
                    return bytes(out)
                out = bytearray(count * 64)
                for i, (x, y) in enumerate(pts):
                    out[i * 64 : i * 64 + 32] = x.to_bytes(32, "big")
                    out[i * 64 + 32 : (i + 1) * 64] = y.to_bytes(32, "big")
                return bytes(out)

        backend = _Ref()

    elif name == "cpu":
        from .cpu import CPUBackend

        backend = CPUBackend()

    elif name == "gpu":
        from .gpu import GPUBackend

        backend = GPUBackend(
            device=opts.get("gpu_device", 0),
            keys_per_thread=opts.get("keys_per_thread", 1024),
            chunk=opts.get("gpu_chunk", 1 << 22),
        )

    else:
        raise ValueError(f"unknown backend {name!r}")

    _BACKEND_CACHE[key] = backend
    return backend


def _resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    from .gpu import is_available as gpu_ok

    if gpu_ok():
        return "gpu"
    # CPU backend needs a C compiler; fall back to the reference otherwise.
    from .cpu import _compiler

    return "cpu" if _compiler() else "reference"


def generate_public_keys(
    start: int,
    count: int,
    compressed: bool = False,
    backend: str = "auto",
    **opts,
) -> bytes:
    """Return public keys for private keys [start, start+count) as raw bytes.

    ``backend`` is one of ``"auto"``, ``"gpu"``, ``"cpu"``, or ``"reference"``.
    Extra keyword options are forwarded to the chosen backend.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    _validate(start, count)
    if count == 0:
        return b""

    resolved = _resolve_backend(backend)
    worker = _make_backend(resolved, opts)
    nthreads = opts.get("nthreads", 0)
    if resolved == "cpu" and nthreads:
        return worker.generate(start, count, compressed, nthreads=nthreads)
    if resolved == "cpu":
        return worker.generate(start, count, compressed)
    return worker.generate(start, count, compressed)


def iter_keys(data: bytes, compressed: bool = False):
    """Yield each public key as an (x, y) tuple of Python ints."""
    if compressed:
        stride = 33
        for i in range(len(data) // stride):
            chunk = data[i * stride : (i + 1) * stride]
            yield chunk[0], int.from_bytes(chunk[1:], "big")
    else:
        stride = 64
        for i in range(len(data) // stride):
            chunk = data[i * stride : (i + 1) * stride]
            yield int.from_bytes(chunk[:32], "big"), int.from_bytes(chunk[32:], "big")


def decode_x_coords(data: bytes, compressed: bool = False):
    """Return the x-coordinates as a numpy array of Python ints (object dtype)."""
    import numpy as np

    if compressed:
        stride = 33
        xs = [int.from_bytes(data[i * stride + 1 : (i + 1) * stride], "big") for i in range(len(data) // stride)]
    else:
        stride = 64
        xs = [int.from_bytes(data[i * stride : i * stride + 32], "big") for i in range(len(data) // stride)]
    return np.array(xs, dtype=object)
