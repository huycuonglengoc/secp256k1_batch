from __future__ import annotations

import random
import unittest

from secp256k1_batch import constants as C
from secp256k1_batch import reference as R
from secp256k1_batch.engine import generate_public_keys, iter_keys


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
