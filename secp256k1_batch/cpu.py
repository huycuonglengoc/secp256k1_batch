"""CPU backend: JIT-compiles the C extension and exposes range generation.

The heavy lifting is done by ``csrc/cpukernel.c``, which is compiled at first
use with the system C compiler (``gcc`` on Colab/Linux, ``clang`` on macOS).
OpenMP is enabled when the compiler supports it so independent sub-ranges run
across cores. The compiled artifact is cached and rebuilt only when the source
changes.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_CSRC_DIR = Path(__file__).parent / "csrc"


def _compiler() -> str | None:
    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _cache_dir() -> Path:
    root = os.environ.get("SECP256K1_BATCH_CACHE")
    base = Path(root) if root else (Path(tempfile.gettempdir()) / "secp256k1_batch_cache")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_library() -> Path:
    source = _CSRC_DIR / "cpukernel.c"
    header = _CSRC_DIR / "secp256k1_point.h"
    field = _CSRC_DIR / "secp256k1_field.h"
    payload = b"".join(p.read_bytes() for p in (source, header, field))
    digest = hashlib.sha256(payload).hexdigest()[:16]
    artifact = _cache_dir() / f"secp256k1_cpu_{digest}.so"
    if artifact.exists():
        return artifact

    cc = _compiler()
    if cc is None:
        raise RuntimeError("No C compiler found; install gcc to use the CPU backend")

    common = ["-O3", "-std=c11", "-fPIC", "-shared"]
    attempts = [
        common + ["-fopenmp"],
        common,
    ]
    last_error: subprocess.CalledProcessError | None = None
    for flags in attempts:
        tmp = artifact.with_suffix(".tmp.so")
        try:
            subprocess.run(
                [cc, *flags, "-I", str(_CSRC_DIR), "-o", str(tmp), str(source)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            last_error = error
            continue
        tmp.replace(artifact)
        return artifact

    raise RuntimeError(f"Failed to compile CPU backend: {last_error}")


def build_generator_table(grp_size: int):
    """Build the affine generator table used by the grouped GPU kernel.

    Returns ``(gx, gy, stride_x, stride_y, stride2_x, stride2_y)`` as
    little-endian ``uint64`` numpy arrays.  ``gx``/``gy`` hold ``grp_size // 2``
    points (``(i+1)*G``); ``stride`` = ``grp_size * G`` and ``stride2`` =
    ``2 * grp_size * G`` (4 limbs each).
    """
    import numpy as np

    lib = ctypes.CDLL(str(_build_library()))
    lib.secp256k1_build_table.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    lib.secp256k1_build_table.restype = ctypes.c_int

    n = grp_size // 2
    gx = np.zeros(n * 4, dtype=np.uint64)
    gy = np.zeros(n * 4, dtype=np.uint64)
    sx = np.zeros(4, dtype=np.uint64)
    sy = np.zeros(4, dtype=np.uint64)
    s2x = np.zeros(4, dtype=np.uint64)
    s2y = np.zeros(4, dtype=np.uint64)
    rc = lib.secp256k1_build_table(
        gx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        gy.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sy.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        s2x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        s2y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_uint64(grp_size),
    )
    if rc != 0:
        raise RuntimeError(f"secp256k1_build_table failed with code {rc}")
    return gx, gy, sx, sy, s2x, s2y


def build_comb_table(start: int, sub: int, hsize: int):
    """Build the comb points the GPU setup kernel needs to compute each thread's
    affine centre without a per-thread scalar multiplication.

    Returns ``(base_x, base_y, jtable)`` as little-endian ``uint64`` numpy
    arrays: ``base`` = ``(start + hsize) * G`` (4 limbs each), ``jtable`` holds
    32 points ``2^k * (sub * G)`` (``x||y``, 8 limbs each).
    """
    import numpy as np

    from .constants import to_limbs

    lib = ctypes.CDLL(str(_build_library()))
    lib.secp256k1_build_comb.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    lib.secp256k1_build_comb.restype = ctypes.c_int

    start_limbs = np.asarray(to_limbs(start), dtype=np.uint64)
    sub_limbs = np.asarray(to_limbs(sub), dtype=np.uint64)
    base_x = np.zeros(4, dtype=np.uint64)
    base_y = np.zeros(4, dtype=np.uint64)
    jtable = np.zeros(32 * 8, dtype=np.uint64)
    rc = lib.secp256k1_build_comb(
        base_x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        base_y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        jtable.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        start_limbs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sub_limbs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_uint64(hsize),
    )
    if rc != 0:
        raise RuntimeError(f"secp256k1_build_comb failed with code {rc}")
    return base_x, base_y, jtable


class CPUBackend:
    """Generates public keys for a contiguous private-key range on the CPU."""

    def __init__(self) -> None:
        self._lib = ctypes.CDLL(str(_build_library()))
        self._lib.secp256k1_pubkeys_range.argtypes = [
            ctypes.c_char_p,  # start (32-byte BE)
            ctypes.c_uint64,  # count
            ctypes.c_char_p,  # output
            ctypes.c_int,     # compressed (0/1)
            ctypes.c_int,     # nthreads
        ]
        self._lib.secp256k1_pubkeys_range.restype = ctypes.c_int

    @property
    def name(self) -> str:
        return "cpu"

    def generate(self, start: int, count: int, compressed: bool = False, nthreads: int = 0) -> bytes:
        if count == 0:
            return b""
        stride = 33 if compressed else 64
        out = ctypes.create_string_buffer(count * stride)
        rc = self._lib.secp256k1_pubkeys_range(
            start.to_bytes(32, "big"), count, out, 1 if compressed else 0, nthreads
        )
        if rc != 0:
            raise RuntimeError(f"CPU backend failed with code {rc}")
        return out.raw
