# secp256k1_batch — high-throughput public-key generation (CPU + GPU)

Generate **millions of secp256k1 public keys** from a contiguous private-key
range, squeezing the last drop of performance out of both the CPU and the GPU.
Designed to run unmodified in **Google Colab**.

This package is built on the design principles of a heterogeneous CPU/GPU
execution pipeline (batching, asynchronous device use, and correctness
validation against a trusted oracle) and applies them to a real cryptographic
workload: deriving `k·G` for the secp256k1 curve used by Bitcoin.

```
private keys:  start, start+1, ..., start+count-1
public keys:   start·G, (start+1)·G, ..., (start+count-1)·G
```

## The single most important optimization

A naive implementation does a full 256-bit scalar multiplication for every key
(~256 point doublings + ~128 additions). Instead, this package computes **only
the first key** of each sub-range with a scalar multiplication and derives every
following key with **one point addition per key** (`P ← P + G`):

* **Jacobian coordinates** — no field inversion in the hot loop.
* **Mixed addition** (`madd-2004-hmv`, 7M+4S) — adds the affine generator to a
  Jacobian accumulator in ~11 field multiplications.
* **Montgomery batch inversion** — converts all Jacobian points back to affine
  with a *single* inversion per sub-range instead of one per key.
* **256-bit modular arithmetic on 64×64→128-bit products** (`unsigned __int128`
  on CPU, the equivalent on CUDA), with the secp256k1-specific reduction
  `2²⁵⁶ ≡ 2³² + 977 (mod p)`.

The same field/point code (`csrc/secp256k1_field.h`, `csrc/secp256k1_point.h`)
is compiled for **both** the CPU and the GPU, so one implementation is
cross-validated by two independent devices against the pure-Python reference.

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

* GPU: `--keys-per-thread` controls the scalar-multiply vs. add trade-off.
  Larger values amortize the per-thread scalar multiplication but reduce
  parallelism. `1024` is a good default for a T4.
* GPU: `--gpu-chunk` bounds device memory; the range is processed in chunks.
* CPU: `--nthreads` sets OpenMP threads (Colab: use the CPU core count).

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
    cpukernel.c         CPU range generator (OpenMP, batch inversion)
    cuda_kernel.cu      GPU range generator (one thread per sub-range)
```
