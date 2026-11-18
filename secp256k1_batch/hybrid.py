"""Hybrid CPU+GPU backend: both devices stay busy at the tasks they do best.

The work is decomposed by sub-problem, not by convenience:

  CPU (host)                            GPU (device)
  ------------------------------------  --------------------------------------
  build the generator table  i*G         setup kernel: comb-add the centres
  build the comb table  2^k*(sub*G)      main kernel: grouped-affine key
  (few, latency-bound scalar mults)      generation (massively parallel)
  split/schedule the range
  SEC-compress chunk N-1 while the
  GPU computes/copies chunk N
  generate its own contiguous share
  of the range (OpenMP, GIL released)
  assemble the final bytes

Concretely, ``generate`` runs the CPU share of the range in a worker thread
while the main thread drives the GPU share through its pipelined path (async
D2H + host-side compression of the previous chunk).  The split is calibrated
once from measured throughput so both devices finish at roughly the same time:

    t_cpu = n_cpu / rate_cpu      t_gpu = n_gpu / rate_gpu
    n_cpu / rate_cpu = n_gpu / rate_gpu   ->   n_cpu = n * rate_cpu/(rate_cpu+rate_gpu)

If no GPU is available the backend degrades to pure CPU generation, which keeps
``backend="hybrid"`` usable (and testable) on CPU-only machines.
"""

from __future__ import annotations

import os
import threading
import time

# Below this many keys the split costs more than it saves (thread spawn, extra
# table/setup work); let the GPU do the whole range.
_MIN_SPLIT_COUNT = 1 << 18  # 262144 keys


def split_counts(count: int, cpu_share: float) -> tuple[int, int]:
    """Return ``(n_cpu, n_gpu)`` with ``n_cpu + n_gpu == count``."""
    share = min(max(float(cpu_share), 0.0), 1.0)
    n_cpu = int(round(count * share))
    n_cpu = max(0, min(count, n_cpu))
    return n_cpu, count - n_cpu


class HybridBackend:
    """Generates keys on the CPU and the GPU at the same time."""

    name = "hybrid"

    def __init__(
        self,
        cpu_share: float | None = None,
        cpu_threads: int = 0,
        calibrate: bool = True,
        gpu=None,
        **gpu_opts,
    ) -> None:
        from .cpu import CPUBackend

        self.cpu = CPUBackend()
        # Leave at least one core free for the main thread, which is busy with
        # the pinned D2H bookkeeping and the output copies while the CPU share
        # runs: oversubscribing every core makes both slower (measured on
        # Colab: 11 workers on 12 vCPUs hurt the hybrid rate).
        self.cpu_threads = int(cpu_threads) or max(1, (os.cpu_count() or 2) - 2)

        if gpu is None:
            from .gpu import GPUBackend, is_available

            # Only a genuinely unavailable device (no CuPy / no GPU) turns this
            # into a CPU-only backend.  A real GPU error (e.g. the kernel fails
            # to compile) must surface, not silently fall back to slow CPU.
            gpu = GPUBackend(**gpu_opts) if is_available() else None
        self.gpu = gpu

        if cpu_share is not None:
            share = float(cpu_share)
        elif self.gpu is None:
            share = 1.0
        elif calibrate:
            share = self._calibrate()
        else:
            share = 0.02

        # Never give the CPU more than a quarter of the range: the GPU is far
        # faster at bulk EC generation, and the host threads also have to carry
        # the D2H copies and output assembly, so an aggressive CPU share would
        # only add memory-bandwidth contention (measured on Colab: a 24% share
        # made the 50M-key run slower than GPU-only).
        limit = 1.0 if self.gpu is None else 0.25
        self.cpu_share = min(max(share, 0.0), limit)

    # -- calibration -----------------------------------------------------

    def _calibrate(self) -> float:
        """Measure both devices once and return the CPU share of the range."""
        try:
            n_cpu, n_gpu = 100_000, 2_000_000
            t0 = time.perf_counter()
            self.cpu.generate(1, n_cpu, nthreads=self.cpu_threads)
            cpu_rate = n_cpu / max(time.perf_counter() - t0, 1e-9)

            t0 = time.perf_counter()
            self.gpu.generate(1, n_gpu)  # end-to-end: kernels + copy + assemble
            gpu_rate = n_gpu / max(time.perf_counter() - t0, 1e-9)

            if cpu_rate <= 0 or gpu_rate <= 0:
                return 0.0
            return cpu_rate / (cpu_rate + gpu_rate)
        except Exception:
            return 0.02

    # -- generation ------------------------------------------------------

    def generate(self, start: int, count: int, compressed: bool = False) -> bytes:
        if count == 0:
            return b""
        if self.gpu is None or self.cpu_share <= 0.0:
            if self.gpu is None:
                return self.cpu.generate(
                    start, count, compressed, nthreads=self.cpu_threads
                )
            return self.gpu.generate(start, count, compressed)
        if count < _MIN_SPLIT_COUNT:
            return self.gpu.generate(start, count, compressed)

        n_cpu, n_gpu = split_counts(count, self.cpu_share)
        if n_cpu == 0 or n_gpu == 0:
            return self.gpu.generate(start, count, compressed)

        # CPU takes the first part, GPU the rest: results stay contiguous and the
        # GPU (the long-running side) starts immediately at its own range.
        result: dict = {}

        def _cpu_worker() -> None:
            try:
                result["data"] = self.cpu.generate(
                    start, n_cpu, compressed, nthreads=self.cpu_threads
                )
            except BaseException as exc:  # propagate to the caller
                result["error"] = exc

        worker = threading.Thread(target=_cpu_worker, name="secp256k1-cpu-share", daemon=True)
        worker.start()
        try:
            gpu_part = self.gpu.generate(start + n_cpu, n_gpu, compressed)
        finally:
            worker.join()

        if "error" in result:
            raise result["error"]
        return result["data"] + gpu_part

    def iter_chunks(
        self,
        start: int,
        count: int,
        compressed: bool = False,
        chunk: int | None = None,
    ):
        """Yield the range in contiguous chunks (constant host memory).

        The CPU share is produced first (it is a contiguous prefix), then the
        GPU streams its own chunks one at a time.
        """
        if count == 0:
            return
        if self.gpu is None or self.cpu_share <= 0.0:
            if self.gpu is None:
                yield self.cpu.generate(
                    start, count, compressed, nthreads=self.cpu_threads
                )
            else:
                yield from self.gpu.iter_chunks(start, count, compressed, chunk)
            return

        n_cpu, n_gpu = split_counts(count, self.cpu_share)
        if n_cpu:
            yield self.cpu.generate(start, n_cpu, compressed, nthreads=self.cpu_threads)
        if n_gpu:
            yield from self.gpu.iter_chunks(start + n_cpu, n_gpu, compressed, chunk)
