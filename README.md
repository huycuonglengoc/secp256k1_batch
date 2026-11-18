# secp256k1_batch — high-throughput public-key generation (CPU + GPU)

Generate **millions of secp256k1 public keys** from a contiguous private-key
range, squeezing the last drop of performance out of both the CPU and the GPU.
Designed to run unmodified in **Google Colab**.

This package is built on the design principles of this repository's
`hetero_pipeline` (heterogeneous CPU/GPU execution, batching, correctness
validation against a trusted oracle), but adds a real cryptographic workload:
deriving `k·G` for the secp256k1 curve used by Bitcoin.

```
private keys:  start, start+1, ..., start+count-1
public keys:   start·G, (start+1)·G, ..., (start+count-1)·G
```

## The single most important optimization

A naive implementation does a full 256-bit scalar multiplication for every key
(~256 point doublings + ~128 additions). This package instead uses the
**grouped-affine** technique — the same algorithm used by high-throughput
scanners such as VanitySearch / KeyHunt (which reach ~1 Gkeys/s on an L4 and
~3.6 Gkeys/s on an RTX 4090):

* **Affine coordinates + shared inversion.** Each thread generates a block of
  `grp_size` keys around an affine centre `S`. The only field inversions are the
  deltas `(d·G).x − S.x`, batch-inverted once per block with a Montgomery
  product tree, and **every inverse is reused by `S − d·G` and `S + d·G`**. Each
  key then costs just **one multiplication + one squaring + a few subtractions**
  (≈2–3 field multiplications), versus ~11 for a Jacobian mixed addition.
* **No per-key scalar multiplication.** Only the block centres need a scalar
  multiplication, and even those are computed with a **comb method**: the host
  builds a 32-point table `2^k·(sub·G)` once, and each thread's centre is at
  most 32 (avg ~16) point additions instead of a full 256-doubling scalar
  multiplication.
* **Register-only field arithmetic.** The 512-bit multiply is a fully unrolled
  row method with all eight accumulators in registers (no local-memory arrays);
  nvcc/NVRTC lower the `unsigned __int128` multiply-accumulate to native
  `mul.lo`/`mul.hi` + `add.cc`/`addc` SASS, with the secp256k1 reduction
  `2²⁵⁶ ≡ 2³² + 977 (mod p)`.
* **Addition-chain inversion.** `secp_inv` uses Brian Smith's addition chain for
  `p − 2` (~267 squarings + 15 multiplications, ≈2.2× faster than Fermat).

The correctness-critical field/point helpers (`csrc/secp256k1_field.h`,
`csrc/secp256k1_point.h`, `csrc/secp256k1_grouped.h`) are compiled for **both**
the CPU and the GPU and cross-validated against the pure-Python reference and a
CPU mirror of the exact grouped algorithm (including the comb setup and the
stride-collision edge case).

## Backends

| Backend      | What it uses                                      | When to use                      |
|--------------|---------------------------------------------------|----------------------------------|
| `gpu`        | CUDA kernel JIT-compiled with CuPy (NVRTC)        | large ranges, a GPU is attached  |
| `cpu`        | C extension JIT-compiled with `gcc`, optional OpenMP | large ranges, CPU-only runtime   |
| `reference`  | pure Python (the correctness oracle)              | validation, tiny ranges          |
| `hybrid`     | CPU **and** GPU concurrently (see below)          | maximum end-to-end throughput    |
| `auto`       | `hybrid` when a GPU is present, else CPU → reference | default                       |

## Hybrid CPU+GPU: which sub-problem runs where

The work is decomposed by sub-problem, and each piece runs on the device that
does it best:

| Sub-problem | Device | Why |
|---|---|---|
| Generator table `i·G`, comb table `2^k·(sub·G)` | **CPU** | a handful of latency-bound scalar multiplications — launching kernels for this would only add overhead |
| Range split / scheduling / calibration | **CPU** | control logic, microseconds of work |
| SEC compression + output assembly of chunk `N-1` | **CPU** (numpy) | memory-bound and highly vectorizable; it runs *while* the GPU works on chunk `N` |
| Its calibrated share of the key range (OpenMP, GIL released) | **CPU** | owns otherwise-idle cores; the share is set so both devices finish together |
| Bulk grouped-affine key generation | **GPU** | massively parallel field arithmetic |

Two pipelines hide the serial costs:

* **GPU side** — chunk `N`'s kernels run on one CUDA stream while chunk `N-1`'s
  device-to-host copy streams into pinned host memory on the other. Any driver
  or CuPy incompatibility falls back to the simple sequential path, so
  correctness never depends on the async machinery.
* **CPU/GPU overlap** — the CPU share of the range runs in a worker thread
  (ctypes releases the GIL) while the main thread drives the GPU. Both shares
  are written into **one** host buffer (the GPU's D2H copies land at the CPU
  share's offset), so the final `tobytes()` is the only pass over the whole
  output — a `cpu_bytes + gpu_bytes` concatenation would copy everything again
  and would cost more than the CPU contributes.

The CPU share is calibrated once (`~100k` CPU keys, `2M` GPU keys) and is
`rate_cpu / (rate_cpu + rate_gpu)`; it is capped at 10% and skipped for ranges
under ~262k keys, where the split costs more than it saves. The cap matters because the output
path is memory-bound: a larger CPU share competes with the D2H copies and with
materialising the returned `bytes`, and becomes the critical path. With no GPU
the backend degrades to pure CPU. Override with `cpu_share=` or
`--cpu-share` when your machine has spare memory bandwidth.

```python
from secp256k1_batch import generate_public_keys
generate_public_keys(1, 10_000_000, backend="hybrid")      # CPU + GPU together
generate_public_keys(1, 10_000_000, backend="hybrid", cpu_share=0.01)  # manual split
```

## Install & run on Google Colab

```bash
# 1. In Colab, choose Runtime → Change runtime type → GPU (T4 or better).
# 2. Upload this folder (or the repo) to /content/sep3.
%cd /content/sep3
!pip -q install cupy-cuda12x   # only needed for the GPU backend
!nvidia-smi                    # confirm a GPU is visible
```

```python
from secp256k1_batch import generate_public_keys, iter_keys

data = generate_public_keys(start=1, count=1_000_000, backend="auto")

# data is count*64 bytes: x || y, 32-byte big-endian each.
for x, y in iter_keys(data):
    if (x, y) != expected: ...
```

Compressed (33-byte SEC) keys:

```python
data = generate_public_keys(start=1, count=100_000, compressed=True)
# 0x02/0x03 parity prefix || x-coordinate
```

Command line:

```bash
python3 -m secp256k1_batch.cli --start 1 --count 1000000 -o pubkeys.hex
python3 -m secp256k1_batch.cli --start 1 --count 1000000 --compressed --backend cpu --nthreads 8 -o pubkeys.hex
```

Streaming (constant RAM, no final full-range copy):

```python
from secp256k1_batch import iter_public_key_chunks

with open("pubkeys.bin", "wb") as handle:
    for part in iter_public_key_chunks(1, 1_000_000_000, backend="hybrid"):
        handle.write(part)
```

`iter_public_key_chunks` yields contiguous chunks (64 B/key uncompressed,
33 B/key compressed) and is what the CLI `-o` path uses; on the GPU backend the
chunks are copied into pinned host buffers asynchronously, so the host never
holds more than one chunk at a time.

## Tuning

* GPU `--grp-size`: keys generated per thread per step (even). Larger values
  amortize the single field inversion over more keys and reduce work per key,
  but use more local memory (at 1024 the per-thread arrays spill to DRAM and
  slow down). `128` is the default; sweep `64`–`2048` for your GPU.
* GPU `--gpu-threads`: target thread count (grid size). The kernel splits the
  range into blocks of `grp_size` keys across up to this many threads.
* GPU `--gpu-chunk`: bounds device memory; the range is processed in chunks.
* CPU `--nthreads`: sets OpenMP threads (Colab: use the CPU core count).

Note: `generate_public_keys` returns the keys as bytes, so the end-to-end rate
for large counts is bounded by host memory, not by the kernel. Two things make
that path as cheap as possible:

* **one host pass.** The uncompressed output is staged directly into the buffer
  that becomes the return value (pinned when it fits ~4 GiB / 40% of free RAM),
  so only the final `tobytes()` touches the data. Compressed keys need one
  conversion pass (64 B/key → 33 B/key).
* **async copies.** D2H runs on two alternating CUDA streams, so chunk N's
  kernels overlap chunk N-1's transfer. Measured on Colab: pinned D2H ~12.8 GB/s
  versus ~2.8 GB/s pageable, while a plain host `tobytes()` is ~1.6 GB/s.
* **device-side SEC compression.** With `compressed=True` the
  `secp256k1_compress_kernel` turns 64-byte keys into 33-byte keys on the GPU,
  so 48% less data crosses PCIe and there is no host conversion pass.

That last number is the real ceiling of a bytes-returning API: producing 3.2 GB
of `bytes` costs about 2 s no matter how fast the GPU is. Use
`iter_public_key_chunks` (or the CLI `-o` path) to skip it when streaming.

## Correctness

Every backend is validated against `secp256k1_batch.reference`, a textbook
affine-coordinate implementation in pure Python, and against well-known
vectors (`1·G = G`, `2·G`, `n·G = ∞`). Run the full suite with:

```bash
python3 -m unittest discover -s tests -v
```

## Layout

```
secp256k1_batch/
  constants.py        curve/field constants + limb helpers
  reference.py        pure-Python oracle
  engine.py           backend selection + public API
  cpu.py              JIT-compiles and drives the C extension
  gpu.py              JIT-compiles and drives the CUDA kernel (CuPy)
  cli.py              command-line tool
  csrc/
    secp256k1_field.h   256-bit modular arithmetic (shared CPU/CUDA)
    secp256k1_point.h   Jacobian point arithmetic (shared CPU/CUDA)
    secp256k1_grouped.h grouped-affine helpers (shared CPU/CUDA)
    cpukernel.c         CPU range generator + generator-table builder
    cuda_kernel.cu      GPU range generator (Jacobian, one thread per sub-range)
    cuda_kernel_grouped.cu  GPU range generator (grouped-affine, fast path)
```
