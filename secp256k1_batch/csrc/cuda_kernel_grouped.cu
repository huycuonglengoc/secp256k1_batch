/* CUDA kernel: grouped-affine batched secp256k1 public-key generation.
 *
 * Fast path.  Instead of a Jacobian "P += G" loop (≈11 field multiplications
 * per key), each thread generates a contiguous block of SECP_GRP_SIZE keys
 * around an affine centre S.  The only field inversions are the deltas
 * (d*G).x - S.x, batch-inverted once per block with a Montgomery product tree;
 * every inverse is shared by S - d*G and S + d*G.  Each key then costs one
 * multiplication, one squaring, and a few subtractions.
 *
 *   secp256k1_setup_kernel   : computes each thread's affine centre.
 *   secp256k1_grouped_kernel : generates the keys.
 *
 * The kernel outputs uncompressed keys (x || y, 64 bytes); the host compresses
 * to SEC (33-byte) form if requested.
 */
#include "secp256k1_point.h"
#include "secp256k1_grouped.h"

/* centre_j = base + sum_{k: bit_k(j)=1} jtable[k], in affine coordinates.
 * base = (start + hsize) * G, jtable[k] = 2^k * (sub * G).  No per-thread
 * scalar multiplication: at most 32 mixed point additions (avg ~16). */
extern "C" __global__ void secp256k1_setup_kernel(
    const uint64_t *base, const uint64_t *jtable,
    uint64_t *cx, uint64_t *cy, const uint64_t T) {
    const uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= T) return;

    secp_pt acc;
    secp_pt_set_inf(&acc);
    secp_pt_add_affine(&acc, &acc, base, base + 4);
    for (int k = 0; k < 32; k++) {
        if ((j >> k) & 1ULL) {
            secp_pt_add_affine(&acc, &acc, jtable + 8 * k, jtable + 8 * k + 4);
        }
    }
    uint64_t x[4], y[4];
    secp_pt_to_affine(x, y, &acc);
    for (int w = 0; w < 4; w++) { cx[j * 4 + w] = x[w]; cy[j * 4 + w] = y[w]; }
}

extern "C" __global__ void secp256k1_grouped_kernel(
    const uint64_t steps, uint8_t *out,
    const uint64_t *gx, const uint64_t *gy, const uint64_t *stride,
    uint64_t *cx, uint64_t *cy, const uint64_t T, const uint64_t count) {
    const uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= T) return;

    const uint64_t sub = steps * SECP_GRP_SIZE;
    const uint64_t base = j * sub;

    /* stride layout: [0..3] = stride.x, [4..7] = stride.y,
     *                [8..11] = 2*stride.x, [12..15] = 2*stride.y */
    uint64_t sx[4], sy[4], stx[4], sty[4], s2x[4], s2y[4];
    for (int w = 0; w < 4; w++) {
        sx[w] = cx[j * 4 + w];
        sy[w] = cy[j * 4 + w];
        stx[w] = stride[w];
        sty[w] = stride[4 + w];
        s2x[w] = stride[8 + w];
        s2y[w] = stride[12 + w];
    }

    uint64_t dx[SECP_HSIZE + 1][4];
    uint64_t prefix[SECP_HSIZE + 1][4];

    for (uint64_t s = 0; s < steps; s++) {
        const uint64_t blk = base + s * SECP_GRP_SIZE;
        if (blk >= count) break;

        for (int i = 0; i < SECP_HSIZE; i++)
            secp_sub(dx[i], gx + 4 * i, sx);
        secp_sub(dx[SECP_HSIZE], stx, sx);

        /* The block centre can equal the stride point (delta.x == 0), which
         * would poison the batch inversion.  Replace it with 1 and remember. */
        int stride_zero = secp_is_zero(dx[SECP_HSIZE]);
        if (stride_zero) {
            dx[SECP_HSIZE][0] = 1;
            dx[SECP_HSIZE][1] = dx[SECP_HSIZE][2] = dx[SECP_HSIZE][3] = 0;
        }
        secp_batch_invert(&dx[0][0], &prefix[0][0], SECP_HSIZE + 1);

        /* centre key (offset blk + HSIZE) */
        uint64_t m = blk + SECP_HSIZE;
        if (m < count) {
            secp_write_be32(out, m * 64, sx);
            secp_write_be32(out, m * 64 + 32, sy);
        }

        uint64_t rx[4], ry[4], qx[4], qy[4], nqy[4];
        for (int i = 0; i < SECP_HSIZE; i++) {
            for (int w = 0; w < 4; w++) { qx[w] = gx[4 * i + w]; qy[w] = gy[4 * i + w]; }
            secp_neg(nqy, qy);

            /* positive side: S + (i+1)*G  (offset blk + HSIZE + i + 1) */
            if (i < SECP_HSIZE - 1) {
                secp_affine_add_inv(rx, ry, sx, sy, qx, qy, dx[i]);
                m = blk + SECP_HSIZE + (uint64_t)i + 1;
                if (m < count) {
                    secp_write_be32(out, m * 64, rx);
                    secp_write_be32(out, m * 64 + 32, ry);
                }
            }
            /* negative side: S - (i+1)*G  (offset blk + HSIZE - i - 1) */
            secp_affine_add_inv(rx, ry, sx, sy, qx, nqy, dx[i]);
            m = blk + SECP_HSIZE - (uint64_t)i - 1;
            if (m < count) {
                secp_write_be32(out, m * 64, rx);
                secp_write_be32(out, m * 64 + 32, ry);
            }
        }

        /* advance centre: S += GRP_SIZE*G (or 2*stride if S == stride) */
        if (stride_zero) {
            for (int w = 0; w < 4; w++) { sx[w] = s2x[w]; sy[w] = s2y[w]; }
        } else {
            secp_affine_add_inv(sx, sy, sx, sy, stx, sty, dx[SECP_HSIZE]);
        }
    }
}
