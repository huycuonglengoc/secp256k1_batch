"""GPU backend: grouped-affine batched secp256k1 public-key generation.

The kernel (``csrc/cuda_kernel_grouped.cu``) implements the grouped-affine
algorithm used by high-throughput scanners (VanitySearch / KeyHunt): each
thread generates a contiguous block of keys around an affine centre, batch
inverting the point deltas once per block.  This replaces the Jacobian
``P += G`` hot loop (≈11 field multiplications per key) with ≈2-3 field
multiplications per key, and removes the per-thread scalar multiplication from
the hot path entirely (only the block centres need one).

The kernel compiles at runtime with CuPy's NVRTC; no ``nvcc`` install is
required.  ``grp_size`` is a compile-time parameter, so the module is cached
per ``grp_size`` value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CSRC_DIR = Path(__file__).parent / "csrc"

# Bumped whenever the kernel sources or the NVRTC compile options change.  The
# Colab bootstrap asserts this marker so a stale checkout (failed git update)
# is caught immediately instead of surfacing as a confusing compiler error.
GPU_BUILD_VERSION = 5

# grp_size -> compiled RawModule (NVRTC runs at most once per grp_size).
_COMPILED: dict = {}


def is_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _compiled_module(grp_size: int):
    if grp_size in _COMPILED:
        return _COMPILED[grp_size]
    import cupy as cp

    source = (_CSRC_DIR / "cuda_kernel_grouped.cu").read_text()
    source = f"#define SECP_GRP_SIZE {grp_size}\n" + source
    # NVRTC requires --device-int128 to enable __int128 in device code, and it
    # has no -O flag (it optimizes fully by default; passing -O3 there is an
    # NVRTC_ERROR_INVALID_OPTION).  nvcc does accept -O3.
    nvrtc_options = ("-I" + str(_CSRC_DIR), "--std=c++11", "--device-int128")
    nvcc_options = ("-I" + str(_CSRC_DIR), "--std=c++11", "-O3")
    try:
        module = cp.RawModule(code=source, options=nvrtc_options)
    except Exception:
        module = cp.RawModule(code=source, options=nvcc_options, backend="nvcc")
    _COMPILED[grp_size] = module
    return module


class GPUBackend:
    """Generates public keys for a contiguous private-key range on the GPU."""

    def __init__(
        self,
        device: int = 0,
        grp_size: int = 128,
        threads: int = 32768,
        block: int = 256,
        chunk: int = 1 << 26,
    ) -> None:
        import cupy as cp

        self.cp = cp
        self.grp_size = int(grp_size)
        if self.grp_size < 2 or (self.grp_size & 1):
            raise ValueError("grp_size must be an even integer >= 2")
        self.target_threads = int(threads)
        self.block = int(block)
        self.chunk = max(1, int(chunk))

        with cp.cuda.Device(device):
            pass

        module = _compiled_module(self.grp_size)
        self._setup = module.get_function("secp256k1_setup_kernel")
        self._main = module.get_function("secp256k1_grouped_kernel")
        self._gx, self._gy, self._stride = self._load_tables()
        self._comb_cache: dict = {}

    def _load_tables(self):
        from .cpu import build_generator_table

        gx, gy, sx, sy, s2x, s2y = build_generator_table(self.grp_size)
        stride = np.concatenate([sx, sy, s2x, s2y]).astype(np.uint64)
        cp = self.cp
        return cp.asarray(gx), cp.asarray(gy), cp.asarray(stride)

    def _comb_points(self, start: int, sub: int, hsize: int):
        """Build (and cache) the device comb points for the setup kernel."""
        key = (start, sub, hsize)
        cached = self._comb_cache.get(key)
        if cached is not None:
            return cached
        from .cpu import build_comb_table

        base_x, base_y, jtable = build_comb_table(start, sub, hsize)
        cp = self.cp
        base = cp.asarray(np.concatenate([base_x, base_y]).astype(np.uint64))
        jt = cp.asarray(jtable)
        self._comb_cache = {key: (base, jt)}
        return base, jt

    @property
    def name(self) -> str:
        return "gpu"

    def _pipeline_chunk(self) -> int:
        """Keys per pipelined chunk: bounded by ``self.chunk``, a pinned-host
        friendly cap, and the currently free device memory (two chunks are live
        at once while the previous one is copied/compressed)."""
        chunk = min(self.chunk, 1 << 22)
        try:
            free, _ = self.cp.cuda.runtime.memGetInfo()
            chunk = min(chunk, max(1 << 20, int(free * 0.25) // 64))
        except Exception:
            pass
        return max(1, int(chunk))

    def generate(self, start: int, count: int, compressed: bool = False) -> bytes:
        """Return the keys for [start, start+count) as raw bytes.

        With more than one chunk the device-to-host copies run asynchronously on
        two alternating CUDA streams into pinned host buffers: while chunk N's
        kernels execute on the GPU, the CPU is already turning chunk N-1 into
        bytes (or SEC-compressing it).  Any driver/CuPy incompatibility falls
        back to the simple sequential path, so correctness never depends on the
        async machinery.
        """
        if count == 0:
            return b""
        ranges = _chunk_ranges(count, self._pipeline_chunk())
        if len(ranges) == 1:
            raw = self._generate_chunk(start, count)
            return _compress(raw) if compressed else raw
        try:
            return self._generate_pipelined(start, ranges, compressed)
        except Exception:
            raw = b"".join(
                self._generate_chunk(start + offset, n) for offset, n in ranges
            )
            return _compress(raw) if compressed else raw

    def _generate_pipelined(self, start: int, ranges, compressed: bool) -> bytes:
        cp = self.cp
        streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True))
        slots: list = [None, None]  # (pinned MemoryPointer, numpy view), ping-pong
        parts: list[bytes] = []
        pending = None  # (stream, host_view, nbytes) of the chunk being copied
        for i, (offset, n) in enumerate(ranges):
            nbytes = n * 64
            stream = streams[i & 1]
            with stream:
                dev, extras = self._launch(start + offset, n, keep=True)
                host = self._pinned_host(slots, i & 1, nbytes)
                # Async D2H on this stream, ordered after the kernels above.
                dev.get(out=host, stream=stream)
            if pending is not None:
                parts.append(self._finish_chunk(pending, compressed))
            # Keep every device buffer of this chunk (`dev`, and the centre
            # arrays / comb points in `extras`) referenced until its stream has
            # been synchronized: CuPy's memory pool is not stream-aware and
            # would otherwise hand the blocks to the next chunk while kernels
            # or the async copy are still reading them.
            pending = (stream, host, nbytes, dev, extras)
        if pending is not None:
            parts.append(self._finish_chunk(pending, compressed))
        return b"".join(parts)

    @staticmethod
    def _pinned_host(slots, idx: int, nbytes: int):
        slot = slots[idx]
        if slot is None or slot[0].nbytes < nbytes:
            import ctypes

            import cupy as cp

            mem = cp.cuda.alloc_pinned_memory(nbytes)
            try:
                host = np.frombuffer(mem, dtype=np.uint8, count=nbytes)
            except Exception:  # MemoryPointer without buffer-protocol support
                buf = (ctypes.c_uint8 * nbytes).from_address(int(mem.ptr))
                host = np.frombuffer(buf, dtype=np.uint8)
            slot = (mem, host)
            slots[idx] = slot
        return slot[1][:nbytes]

    @staticmethod
    def _finish_chunk(pending, compressed: bool) -> bytes:
        stream, host, nbytes, _dev, _extras = pending
        stream.synchronize()  # the copy for this chunk is now complete
        view = host[:nbytes]
        return _compress(view) if compressed else view.tobytes()

    def _generate_chunk(self, start: int, n: int) -> bytes:
        out = self._launch(start, n)
        return out.get().tobytes()

    def _launch(self, start: int, n: int, keep: bool = False):
        """Run setup + main kernels, returning the (uncompressed) device array.

        With ``keep=True`` also return the working device buffers (centres and
        comb points) so a pipelined caller can keep them alive until the stream
        has been synchronized.
        """
        cp = self.cp
        grp = self.grp_size
        hsize = grp // 2
        blocks = (n + grp - 1) // grp
        T = min(blocks, self.target_threads)
        steps = (blocks + T - 1) // T
        sub = steps * grp

        out = cp.empty(n * 64, dtype=cp.uint8)
        cx = cp.empty(T * 4, dtype=cp.uint64)
        cy = cp.empty(T * 4, dtype=cp.uint64)

        base, jtable = self._comb_points(start, sub, hsize)
        grid = (T + self.block - 1) // self.block
        block = self.block

        self._setup(
            (grid,),
            (block,),
            (base, jtable, cx, cy, np.uint64(T)),
        )
        self._main(
            (grid,),
            (block,),
            (np.uint64(steps), out, self._gx, self._gy, self._stride,
             cx, cy, np.uint64(T), np.uint64(n)),
        )
        if keep:
            return out, (cx, cy, base, jtable)
        return out

    def benchmark_kernel(self, start: int, count: int, iterations: int = 3) -> float:
        """Return the best setup+main kernel time in seconds (GPU compute only).

        This excludes the device-to-host transfer, which dominates the
        end-to-end ``generate`` time for large ``count``.
        """
        cp = self.cp
        if count == 0:
            return 0.0
        # warm up (also forces table upload + any lazy JIT)
        self._launch(start, count)
        cp.cuda.get_current_stream().synchronize()

        best = float("inf")
        for _ in range(iterations):
            start_ev = cp.cuda.Event()
            end_ev = cp.cuda.Event()
            start_ev.record()
            self._launch(start, count)
            end_ev.record()
            end_ev.synchronize()
            best = min(best, cp.cuda.get_elapsed_time(start_ev, end_ev) / 1000.0)
        return best


def _chunk_ranges(count: int, chunk: int):
    """Split [0, count) into ``(offset, n)`` ranges of at most ``chunk`` keys."""
    return [(offset, min(chunk, count - offset)) for offset in range(0, count, chunk)]


def _compress(uncompressed) -> bytes:
    """Convert uncompressed (x || y, 64 bytes) to SEC compressed (33 bytes).

    Accepts either ``bytes`` or a uint8 ``numpy.ndarray`` view (the pipelined
    path hands us a pinned-host view, so no extra copy is needed).
    """
    if len(uncompressed) == 0:
        return b""
    if isinstance(uncompressed, np.ndarray):
        arr = uncompressed.reshape(-1, 64)
    else:
        arr = np.frombuffer(uncompressed, dtype=np.uint8).reshape(-1, 64)
    n = arr.shape[0]
    out = np.empty((n, 33), dtype=np.uint8)
    out[:, 0] = 0x02 | (arr[:, 63] & 1)
    out[:, 1:] = arr[:, :32]
    return out.tobytes()
