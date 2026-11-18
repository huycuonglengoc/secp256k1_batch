/* CPU backend for batched secp256k1 public-key generation.
 *
 * Given a starting private key and a count, it produces the public keys for
 * the contiguous range [start, start+count). Only the first key of each
 * thread's sub-range is computed with a full scalar multiplication; the rest
 * use a single mixed point addition (P += G) per key. Field inversions are
 * amortized with Montgomery batch inversion over the whole sub-range.
 *
 * The heavy inner loop is plain C with `unsigned __int128` products, so GCC
 * vectorizes/optimizes it well; OpenMP spreads independent sub-ranges across
 * CPU cores when compiled with -fopenmp.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "secp256k1_point.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- byte/limb conversions ---------------------------------------------- */

static void be32_to_limbs(uint64_t limbs[4], const uint8_t *be32) {
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) v = (v << 8) | be32[24 - 8 * i + b];
        limbs[i] = v;
    }
}

static void limbs_to_be32(uint8_t *be32, const uint64_t limbs[4]) {
    for (int i = 0; i < 4; i++) {
        uint64_t v = limbs[i];
        for (int b = 7; b >= 0; b--) {
            be32[24 - 8 * i + b] = (uint8_t)(v & 0xFF);
            v >>= 8;
        }
    }
}

/* Add a 64-bit offset to a 256-bit little-endian limb scalar (with carry). */
static void scalar_add_u64(uint64_t s[4], uint64_t offset) {
    unsigned __int128 carry = offset;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 t = (unsigned __int128)s[i] + (uint64_t)carry;
        s[i] = (uint64_t)t;
        carry = t >> 64;
    }
}

/* ---- Montgomery batch inversion ------------------------------------------ */

/* Invert z[0..count-1] in place (each z_i is 4 limbs). work must hold
 * count * 8 uint64_t. Requires every z_i != 0. */
static void batch_invert_impl(uint64_t *z, uint64_t *work, size_t count) {
    if (count == 0) return;
    uint64_t *prefix = work;               /* count * 4 limbs */
    uint64_t *orig = work + count * 4;     /* second half of work */
    memcpy(orig, z, count * 4 * sizeof(uint64_t));

    memcpy(prefix, z, 4 * sizeof(uint64_t));
    for (size_t i = 1; i < count; i++) {
        secp_mul(prefix + 4 * i, prefix + 4 * (i - 1), orig + 4 * i);
    }

    uint64_t inv[4];
    secp_inv(inv, prefix + 4 * (count - 1));

    uint64_t u[4];
    memcpy(u, inv, sizeof(u));
    for (size_t i = count; i-- > 0;) {
        if (i == 0) {
            memcpy(z + 4 * i, u, sizeof(u));
        } else {
            secp_mul(z + 4 * i, u, prefix + 4 * (i - 1));
            secp_mul(u, u, orig + 4 * i);
        }
    }
}

/* ---- core generation ------------------------------------------------------ */

/* Generate `count` public keys for private keys [start, start+count).
 *   start_be : 32-byte big-endian starting private key (in [1, n-1]).
 *   out      : buffer of count * 64 bytes (uncompressed) or count * 33 (compressed).
 *   compressed: 0 -> 64-byte (x||y), 1 -> 33-byte (0x02/0x03 || x).
 *   nthreads : OpenMP threads (<=0 -> single-threaded).
 * Returns 0 on success, non-zero on invalid input. */
int secp256k1_pubkeys_range(const uint8_t *start_be, uint64_t count,
                            uint8_t *out, int compressed, int nthreads) {
    if (count == 0) return 0;
    if (nthreads < 1) nthreads = 1;

    uint64_t start[4];
    be32_to_limbs(start, start_be);

    const size_t out_stride = compressed ? 33 : 64;

#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

    uint64_t threads = (uint64_t)nthreads;
    if (threads > count) threads = count;

    int rc = 0;

#pragma omp parallel for schedule(static) num_threads((int)threads)
    for (uint64_t t = 0; t < threads; t++) {
        uint64_t begin = (count / threads) * t + (t < count % threads ? t : count % threads);
        uint64_t length = count / threads + (t < count % threads ? 1 : 0);
        if (length == 0) continue;

        uint64_t base[4] = {start[0], start[1], start[2], start[3]};
        scalar_add_u64(base, begin);

        secp_pt *pts = (secp_pt *)malloc((size_t)length * sizeof(secp_pt));
        if (!pts) { rc = 2; continue; }

        secp_pt acc;
        secp_scalar_mult(&acc, base);
        for (uint64_t i = 0; i < length; i++) {
            pts[i] = acc;
            secp_pt_add_generator(&acc, &acc);
        }

        /* batch-invert all z coordinates */
        uint64_t *zs = (uint64_t *)malloc((size_t)length * 4 * sizeof(uint64_t));
        uint64_t *work = (uint64_t *)malloc((size_t)length * 8 * sizeof(uint64_t));
        if (!zs || !work) { free(zs); free(work); free(pts); rc = 2; continue; }
        for (uint64_t i = 0; i < length; i++)
            memcpy(zs + 4 * i, pts[i].z, 4 * sizeof(uint64_t));
        batch_invert_impl(zs, work, length);

        for (uint64_t i = 0; i < length; i++) {
            uint64_t x[4], y[4];
            uint64_t zinv[4], zinv2[4], zinv3[4];
            memcpy(zinv, zs + 4 * i, 4 * sizeof(uint64_t));
            secp_sqr(zinv2, zinv);
            secp_mul(zinv3, zinv2, zinv);
            secp_mul(x, pts[i].x, zinv2);
            secp_mul(y, pts[i].y, zinv3);
            uint8_t *dst = out + (size_t)(begin + i) * out_stride;
            if (compressed) {
                dst[0] = (uint8_t)(0x02 | (y[0] & 1));
                limbs_to_be32(dst + 1, x);
            } else {
                limbs_to_be32(dst, x);
                limbs_to_be32(dst + 32, y);
            }
        }

        free(zs);
        free(work);
        free(pts);
    }

    return rc;
}

/* Reference single scalar multiplication used by the self-test only. */
void secp256k1_pubkey_single(const uint8_t *k_be, uint8_t *out, int compressed) {
    uint64_t k[4];
    be32_to_limbs(k, k_be);
    secp_pt p;
    secp_scalar_mult(&p, k);
    uint64_t x[4], y[4];
    secp_pt_to_affine(x, y, &p);
    if (compressed) {
        out[0] = (uint8_t)(0x02 | (y[0] & 1));
        limbs_to_be32(out + 1, x);
    } else {
        limbs_to_be32(out, x);
        limbs_to_be32(out + 32, y);
    }
}
