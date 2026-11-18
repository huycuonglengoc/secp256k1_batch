"""GPU backend: compiles the CUDA kernel at runtime with CuPy and executes it.

The kernel is in ``csrc/cuda_kernel.cu`` and shares the exact field/point
arithmetic with the CPU backend (``secp256k1_field.h`` / ``secp256k1_point.h``),
so one implementation is exercised by both devices. CuPy's NVRTC compiles the
source on the fly; no separate ``nvcc`` install is required.
"""

from __future__ import annotations

from pathlib import Path

from .constants import MASK64, to_limbs

_CSRC_DIR = Path(__file__).parent / "csrc"

# Compiled CUDA module, cached so NVRTC runs at most once per process. The
# kernel source does not depend on device/keys_per_thread/chunk (those are
# passed as kernel arguments), so one module is shared by every backend.
_COMPILED_MODULE = None


def is_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _compiled_module():
    """Compile (once) and return the CuPy RawModule for the CUDA kernel."""
    global _COMPILED_MODULE
    if _COMPILED_MODULE is None:
        import cupy as cp

        source = (_CSRC_DIR / "cuda_kernel.cu").read_text()
        # NVRTC requires --device-int128 to enable __int128 in device code; nvcc
        # supports it by default and does not accept that flag.
        nvrtc_options = ("-I" + str(_CSRC_DIR), "--std=c++11", "--device-int128")
        nvcc_options = ("-I" + str(_CSRC_DIR), "--std=c++11")
        try:
            _COMPILED_MODULE = cp.RawModule(code=source, options=nvrtc_options)
        except Exception:
            _COMPILED_MODULE = cp.RawModule(code=source, options=nvcc_options, backend="nvcc")
    return _COMPILED_MODULE


class GPUBackend:
    """Generates public keys for a contiguous private-key range on the GPU."""

    def __init__(self, device: int = 0, keys_per_thread: int = 64, chunk: int = 1 << 22) -> None:
        import cupy as cp

        self.cp = cp
        self.keys_per_thread = keys_per_thread
        self.chunk = chunk
        try:
            with cp.cuda.Device(device):
                pass
        except Exception as error:
            raise RuntimeError(f"CUDA device {device} unavailable: {error}") from error

        self._kernel = _compiled_module().get_function("secp256k1_batch_kernel")

    @property
    def name(self) -> str:
        return "gpu"

    def generate(self, start: int, count: int, compressed: bool = False) -> bytes:
        chunks: list[bytes] = []
        for offset in range(0, count, self.chunk):
            n = min(self.chunk, count - offset)
            chunks.append(self._generate_chunk(start + offset, n))
        raw = b"".join(chunks)
        if compressed:
            return _compress(raw)
        return raw

    def _generate_chunk(self, start: int, n: int) -> bytes:
        if n == 0:
            return b""
        import numpy as np

        cp = self.cp
        u64 = np.uint64
        kpt = self.keys_per_thread
        out = cp.empty(n * 64, dtype=cp.uint8)
        z = cp.empty(n * 4, dtype=cp.uint64)
        prefix = cp.empty(n * 4, dtype=cp.uint64)

        s0, s1, s2, s3 = to_limbs(start)
        num_threads = (n + kpt - 1) // kpt
        block = 256
        grid = (num_threads + block - 1) // block
        self._kernel(
            (grid,),
            (block,),
            (u64(s0), u64(s1), u64(s2), u64(s3), u64(kpt), out, z, prefix, u64(n)),
        )
        cp.cuda.get_current_stream().synchronize()
        return out.get().tobytes()


def _compress(uncompressed: bytes) -> bytes:
    """Convert uncompressed (x || y, 64 bytes) to SEC compressed (33 bytes)."""
    import numpy as np

    if not uncompressed:
        return b""
    arr = np.frombuffer(uncompressed, dtype=np.uint8).reshape(-1, 64)
    n = arr.shape[0]
    out = np.empty((n, 33), dtype=np.uint8)
    out[:, 0] = 0x02 | (arr[:, 63] & 1)
    out[:, 1:] = arr[:, :32]
    return out.tobytes()
