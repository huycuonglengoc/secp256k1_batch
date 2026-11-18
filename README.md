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
* **PTX inline-asm field arithmetic.** The 64×64→128-bit multiply-accumulate hot
  path uses native `mul.lo.u64`/`mul.hi.u64`/`add.cc`/`addc.cc` PTX (the same
  technique as KeyHunt's `GPUMath.h`), with the secp256k1 reduction
  `2²⁵⁶ ≡ 2³² + 977 (mod p)`.
* **Addition-chain inversion and symmetric squaring.** `secp_inv` uses Brian
  Smith's addition chain for `p − 2` (~267 squarings + 15 multiplications, ≈2.2×
  faster than Fermat), and `secp_sqr` exploits the `aᵢaⱼ = aⱼaᵢ` symmetry
  (10 products instead of 16).

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
| `auto`       | GPU → CPU → reference, whichever is available      | default                          |

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
for very large counts is ultimately bounded by PCIe/device-to-host transfer,
not by the kernel. The kernel itself runs substantially faster than the
transfer; benchmark it directly with CUDA events if you want the raw GPU rate.

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
