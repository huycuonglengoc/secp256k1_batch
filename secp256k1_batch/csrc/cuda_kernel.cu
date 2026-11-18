/* CUDA kernel for batched secp256k1 public-key generation.
 *
 * One thread generates a contiguous sub-range of keys: it computes the base
 * point with a single scalar multiplication, then walks the range with a
 * mixed point addition (P += G) per key. Field inversions are amortized with a
 * Montgomery batch inversion over the thread's whole sub-range.
 *
 * Memory layout (all device global memory, little-endian uint64 limbs):
 *   out    : count * 64 bytes. During compute it holds Jacobian X||Y; at the
 *            end it is overwritten with the affine public key (x||y big-endian).
 *   z      : count * 4 uint64, holds Z then the batch-inverted Z.
 *   prefix : count * 4 uint64, scratch for the batch inversion.
 *
 * This kernel intentionally outputs uncompressed keys; the host compresses to
 * SEC (33-byte) form if requested.
 */
#include "secp256k1_point.h"

extern "C" __global__ void secp256k1_batch_kernel(
    const uint64_t start0, const uint64_t start1, const uint64_t start2,
    const uint64_t start3, const uint64_t keys_per_thread, uint8_t *out,
    uint64_t *z, uint64_t *prefix, const uint64_t count) {
    const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t begin = t * keys_per_thread;
    if (begin >= count) return;
    const uint64_t length = (keys_per_thread < count - begin) ? keys_per_thread
                                                              : (count - begin);

    /* base = start + begin (64-bit offset, propagated with carry). */
    uint64_t base[4] = {start0, start1, start2, start3};
    unsigned __int128 carry = begin;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 s = (unsigned __int128)base[i] + (uint64_t)carry;
        base[i] = (uint64_t)s;
        carry = s >> 64;
    }

    secp_pt acc;
    secp_scalar_mult(&acc, base);

    /* Walk the range, storing Jacobian points. */
    for (uint64_t i = 0; i < length; i++) {
        const uint64_t idx = begin + i;
        uint64_t *dst = (uint64_t *)out + idx * 8;
        for (int w = 0; w < 4; w++) {
            dst[w] = acc.x[w];
            dst[4 + w] = acc.y[w];
            z[idx * 4 + w] = acc.z[w];
        }
        secp_pt_add_generator(&acc, &acc);
    }

    /* Montgomery batch inversion of z[begin .. begin+length). */
    uint64_t *myz = z + begin * 4;
    uint64_t *myp = prefix + begin * 4;
    for (int w = 0; w < 4; w++) myp[w] = myz[w];
    for (uint64_t i = 1; i < length; i++) {
        secp_mul(myp + i * 4, myp + (i - 1) * 4, myz + i * 4);
    }
    uint64_t inv[4];
    secp_inv(inv, myp + (length - 1) * 4);
    uint64_t u[4] = {inv[0], inv[1], inv[2], inv[3]};
    for (uint64_t ii = length; ii-- > 0;) {
        const uint64_t i = ii;
        if (i == 0) {
            for (int w = 0; w < 4; w++) myz[w] = u[w];
        } else {
            uint64_t zinv[4];
            secp_mul(zinv, u, myp + (i - 1) * 4);
            secp_mul(u, u, myz + i * 4); /* original z[i] */
            for (int w = 0; w < 4; w++) myz[i * 4 + w] = zinv[w];
        }
    }

    /* Convert Jacobian -> affine and write big-endian x || y. */
    for (uint64_t i = 0; i < length; i++) {
        const uint64_t idx = begin + i;
        const uint64_t *src = (const uint64_t *)out + idx * 8;
        uint64_t X[4], Y[4];
        for (int w = 0; w < 4; w++) {
            X[w] = src[w];
            Y[w] = src[4 + w];
        }
        uint64_t zinv[4], zinv2[4], zinv3[4], x[4], yy[4];
        for (int w = 0; w < 4; w++) zinv[w] = myz[i * 4 + w];
        secp_sqr(zinv2, zinv);
        secp_mul(zinv3, zinv2, zinv);
        secp_mul(x, X, zinv2);
        secp_mul(yy, Y, zinv3);
        for (int w = 0; w < 4; w++) {
            uint64_t v = x[w];
            for (int b = 7; b >= 0; b--) {
                out[idx * 64 + (24 - 8 * w) + b] = (uint8_t)(v & 0xFF);
                v >>= 8;
            }
            v = yy[w];
            for (int b = 7; b >= 0; b--) {
                out[idx * 64 + 32 + (24 - 8 * w) + b] = (uint8_t)(v & 0xFF);
                v >>= 8;
            }
        }
    }
}
