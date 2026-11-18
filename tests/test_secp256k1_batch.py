from __future__ import annotations

import random
import unittest

from secp256k1_batch import constants as C
from secp256k1_batch import reference as R
from secp256k1_batch.engine import (
    generate_public_keys,
    iter_keys,
    iter_public_key_chunks,
)


class ReferenceTests(unittest.TestCase):
    def test_known_vectors(self) -> None:
        self.assertEqual(R.public_key(1), (C.GX, C.GY))
        self.assertEqual(R.public_key(2), (C.TWO_G_X, C.TWO_G_Y))
        self.assertIsNone(R.scalar_mult(C.N))
        self.assertEqual(R.public_key(C.N - 1), (C.GX, (-C.GY) % C.P))

    def test_range_matches_scalar_mult(self) -> None:
        start, count = 1234, 50
        self.assertEqual(R.public_keys_range(start, count), [R.public_key(k) for k in range(start, start + count)])

    def test_increment_matches_add(self) -> None:
        p = R.public_key(98765)
        self.assertEqual(R.point_add(p, (C.GX, C.GY)), R.public_key(98766))

    def test_grouped_range_matches(self) -> None:
        for grp in (4, 8, 64, 256):
            for start in (1, 2, 100):
                n = 2000  # large enough to exercise the stride-advance collision
                self.assertEqual(
                    R.public_keys_range_grouped(start, n, grp_size=grp),
                    R.public_keys_range(start, n),
                )


class EngineTests(unittest.TestCase):
    def test_reference_backend_roundtrip(self) -> None:
        start, count = 5, 7
        data = generate_public_keys(start, count, backend="reference")
        self.assertEqual(len(data), count * 64)
        self.assertEqual(list(iter_keys(data)), R.public_keys_range(start, count))

    def test_compressed_reference(self) -> None:
        start, count = 11, 6
        data = generate_public_keys(start, count, compressed=True, backend="reference")
        self.assertEqual(len(data), count * 33)
        pts = R.public_keys_range(start, count)
        for i, (prefix, x) in enumerate(iter_keys(data, compressed=True)):
            self.assertEqual(prefix, 0x02 | (pts[i][1] & 1))
            self.assertEqual(x, pts[i][0])

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            generate_public_keys(0, 5, backend="reference")
        with self.assertRaises(ValueError):
            generate_public_keys(C.N, 5, backend="reference")
        with self.assertRaises(ValueError):
            generate_public_keys(1, -1, backend="reference")

    def test_package_exports(self) -> None:
        # Every advertised name must actually import from the package root
        # (regression: iter_public_key_chunks lived only in engine.py).
        import secp256k1_batch as pkg

        for name in pkg.__all__:
            self.assertTrue(hasattr(pkg, name), f"missing export: {name}")
        self.assertIs(pkg.iter_public_key_chunks, iter_public_key_chunks)

    def test_streaming_chunks_match_full(self) -> None:
        start, count = 3, 100
        full = generate_public_keys(start, count, backend="reference")
        parts = list(iter_public_key_chunks(start, count, backend="reference", chunk=7))
        self.assertEqual(b"".join(parts), full)
        self.assertEqual(len(parts), 15)  # ceil(100 / 7)


class CPUBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from secp256k1_batch.cpu import _compiler

        if _compiler() is None:
            raise unittest.SkipTest("no C compiler available")

    def test_cpu_matches_reference(self) -> None:
        data = generate_public_keys(10, 40, backend="cpu")
        self.assertEqual(list(iter_keys(data)), R.public_keys_range(10, 40))

    def test_cpu_compressed(self) -> None:
        data = generate_public_keys(100, 25, compressed=True, backend="cpu")
        pts = R.public_keys_range(100, 25)
        for i, (prefix, x) in enumerate(iter_keys(data, compressed=True)):
            self.assertEqual(prefix, 0x02 | (pts[i][1] & 1))
            self.assertEqual(x, pts[i][0])

    def test_cpu_multithreaded(self) -> None:
        random.seed(7)
        start = random.randrange(1, C.N - 1000)
        count = 500
        data = generate_public_keys(start, count, backend="cpu", nthreads=4)
        self.assertEqual(list(iter_keys(data)), R.public_keys_range(start, count))

    def test_build_generator_table(self) -> None:
        from secp256k1_batch.cpu import build_generator_table

        grp = 256
        gx, gy, sx, sy, s2x, s2y = build_generator_table(grp)
        n = grp // 2
        for i in range(n):
            p = R.public_key(i + 1)
            xi = sum(int(gx[4 * i + w]) << (64 * w) for w in range(4))
            yi = sum(int(gy[4 * i + w]) << (64 * w) for w in range(4))
            self.assertEqual((xi, yi), p)
        stride = R.public_key(grp)
        stride2 = R.public_key(2 * grp)
        self.assertEqual(
            (sum(int(sx[w]) << (64 * w) for w in range(4)),
             sum(int(sy[w]) << (64 * w) for w in range(4))),
            stride,
        )
        self.assertEqual(
            (sum(int(s2x[w]) << (64 * w) for w in range(4)),
             sum(int(s2y[w]) << (64 * w) for w in range(4))),
            stride2,
        )


class HybridBackendTests(unittest.TestCase):
    """The hybrid split logic is tested with a stub GPU backend (the real one
    needs CUDA), so range partitioning, ordering and concurrency are covered on
    any machine."""

    @classmethod
    def setUpClass(cls) -> None:
        from secp256k1_batch.cpu import _compiler

        if _compiler() is None:
            raise unittest.SkipTest("no C compiler available")

    def setUp(self) -> None:
        from secp256k1_batch import hybrid

        # Exercise the real split path with small ranges (the reference oracle
        # is pure Python, so keep the key counts tiny).
        self._saved_min = hybrid._MIN_SPLIT_COUNT
        hybrid._MIN_SPLIT_COUNT = 100

    def tearDown(self) -> None:
        from secp256k1_batch import hybrid

        hybrid._MIN_SPLIT_COUNT = self._saved_min

    class _StubGPU:
        """Behaves like the GPU backend but generates on the CPU (1 thread)."""

        name = "gpu-stub"

        def __init__(self) -> None:
            from secp256k1_batch.cpu import CPUBackend

            self._cpu = CPUBackend()
            self.calls: list[tuple[int, int, bool]] = []

        def generate(self, start: int, count: int, compressed: bool = False) -> bytes:
            self.calls.append((start, count, compressed))
            return self._cpu.generate(start, count, compressed, nthreads=1)

    def test_split_counts_invariants(self) -> None:
        from secp256k1_batch.hybrid import split_counts

        for count in (0, 1, 7, 1000, 2**20 - 1):
            for share in (0.0, 0.01, 0.25, 0.5, 1.0):
                n_cpu, n_gpu = split_counts(count, share)
                self.assertEqual(n_cpu + n_gpu, count)
                self.assertGreaterEqual(n_cpu, 0)
                self.assertGreaterEqual(n_gpu, 0)
        # share is clamped to [0, 1]
        self.assertEqual(split_counts(1000, -1.0), (0, 1000))
        self.assertEqual(split_counts(1000, 2.0), (1000, 0))

    def test_hybrid_matches_reference(self) -> None:
        from secp256k1_batch.hybrid import HybridBackend, split_counts

        stub = self._StubGPU()
        backend = HybridBackend(gpu=stub, cpu_share=0.5, cpu_threads=2, calibrate=False)
        for start, count in ((1, 1000), (12345, 1001), (0xDEADBEEF, 1000)):
            before = len(stub.calls)
            data = backend.generate(start, count)
            self.assertEqual(len(data), count * 64)
            self.assertEqual(list(iter_keys(data)), R.public_keys_range(start, count))
            # The GPU (stub) got exactly the second contiguous half.
            n_cpu, n_gpu = split_counts(count, backend.cpu_share)
            self.assertEqual(stub.calls[before:], [(start + n_cpu, n_gpu, False)])

    def test_hybrid_compressed_and_small(self) -> None:
        from secp256k1_batch.hybrid import HybridBackend

        stub = self._StubGPU()
        backend = HybridBackend(gpu=stub, cpu_share=0.5, cpu_threads=2, calibrate=False)
        # Small counts bypass the split entirely (GPU does everything).
        data = backend.generate(7, 10, compressed=True)
        self.assertEqual(len(data), 10 * 33)
        # Larger compressed range: split path with 33-byte output.
        data = backend.generate(7, 1005, compressed=True)
        pts = R.public_keys_range(7, 1005)
        for i, (prefix, x) in enumerate(iter_keys(data, compressed=True)):
            self.assertEqual(prefix, 0x02 | (pts[i][1] & 1))
            self.assertEqual(x, pts[i][0])

    def test_hybrid_engine_backend(self) -> None:
        # On machines without a GPU the hybrid backend degrades to CPU-only.
        data = generate_public_keys(11, 300, backend="hybrid", cpu_share=1.0)
        self.assertEqual(list(iter_keys(data)), R.public_keys_range(11, 300))

    def test_chunk_ranges(self) -> None:
        from secp256k1_batch.gpu import _chunk_ranges

        self.assertEqual(_chunk_ranges(0, 10), [])
        self.assertEqual(_chunk_ranges(10, 10), [(0, 10)])
        self.assertEqual(_chunk_ranges(23, 10), [(0, 10), (10, 10), (20, 3)])
        for count, chunk in ((1, 1), (5, 2), (100, 7)):
            ranges = _chunk_ranges(count, chunk)
            self.assertEqual([o for o, _ in ranges], list(range(0, count, chunk)))
            self.assertEqual(sum(n for _, n in ranges), count)


class GPUBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from secp256k1_batch.gpu import is_available

        if not is_available():
            raise unittest.SkipTest("CuPy/CUDA unavailable")

    def test_gpu_matches_reference(self) -> None:
        data = generate_public_keys(20, 128, backend="gpu")
        self.assertEqual(list(iter_keys(data)), R.public_keys_range(20, 128))

    def test_gpu_compressed(self) -> None:
        data = generate_public_keys(50, 64, compressed=True, backend="gpu")
        pts = R.public_keys_range(50, 64)
        for i, (prefix, x) in enumerate(iter_keys(data, compressed=True)):
            self.assertEqual(prefix, 0x02 | (pts[i][1] & 1))
            self.assertEqual(x, pts[i][0])


if __name__ == "__main__":
    unittest.main()
