#!/usr/bin/env python3
"""Command-line tool for batched secp256k1 public-key generation.

Examples:
  # Generate 1,000,000 public keys starting at private key 1 (GPU if available):
  python3 -m secp256k1_batch.cli --start 1 --count 1000000 -o pubkeys.txt

  # Compressed keys on the CPU with 8 threads, hex output:
  python3 -m secp256k1_batch.cli --start 0x1 --count 100000 --compressed \
      --backend cpu --nthreads 8 -o pubkeys.hex --format hex
"""

from __future__ import annotations

import argparse
import sys
import time

from .engine import generate_public_keys


def _parse_int(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_int, help="first private key (decimal or 0x hex)")
    parser.add_argument("--count", required=True, type=int, help="number of keys to generate")
    parser.add_argument("--compressed", action="store_true", help="emit 33-byte SEC compressed keys")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu", "reference"], default="auto")
    parser.add_argument("--nthreads", type=int, default=0, help="CPU OpenMP threads (0 = default)")
    parser.add_argument("--keys-per-thread", type=int, default=1024, help="GPU keys per thread")
    parser.add_argument("--gpu-chunk", type=int, default=1 << 22, help="GPU chunk size (bounds device memory)")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("-o", "--output", help="output file (default: stdout)")
    parser.add_argument("--format", choices=["hex", "bin"], default="hex")
    return parser


def _emit_hex(data: bytes, compressed: bool, handle) -> None:
    stride = 33 if compressed else 64
    for i in range(len(data) // stride):
        handle.write(data[i * stride : (i + 1) * stride].hex() + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 0:
        raise SystemExit("--count must be non-negative")

    started = time.perf_counter()
    data = generate_public_keys(
        args.start,
        args.count,
        compressed=args.compressed,
        backend=args.backend,
        nthreads=args.nthreads,
        keys_per_thread=args.keys_per_thread,
        gpu_chunk=args.gpu_chunk,
        gpu_device=args.gpu_device,
    )
    elapsed = time.perf_counter() - started

    if args.output:
        mode = "wb" if args.format == "bin" else "w"
        with open(args.output, mode) as handle:
            if args.format == "bin":
                handle.write(data)
            else:
                _emit_hex(data, args.compressed, handle)
    else:
        _emit_hex(data, args.compressed, sys.stdout)

    if args.count:
        print(f"{args.count} keys in {elapsed:.3f}s ({args.count / elapsed:.0f} keys/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
