/* Grouped-affine range generation, shared by the CPU reference and the CUDA
 * kernel.
 *
 * Instead of a Jacobian "P += G" loop (≈11 field multiplications per key), one
 * thread produces a contiguous block of SECP_GRP_SIZE keys centred on an affine
 * point S.  The keys S - d*G and S + d*G share the modular inverse of
 * (d*G).x - S.x, so a single Montgomery batch inversion covers the whole group
 * and every key then costs one multiplication, one squaring, and a few
 * subtractions.  This is the technique used by high-throughput scanners such
 * as VanitySearch / KeyHunt.
 *
 * SECP_GRP_SIZE is a compile-time constant (see cuda_kernel_grouped.cu).
 */
#ifndef SECP256K1_GROUPED_H
#define SECP256K1_GROUPED_H

#include "secp256k1_field.h"

#ifndef SECP_GRP_SIZE
#define SECP_GRP_SIZE 256
#endif
#define SECP_HSIZE (SECP_GRP_SIZE / 2)

/* Group order n, and NC = 2^256 - n (used to reduce a 257-bit scalar mod n). */
SECP_CONST uint64_t SECP_N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL,
};
SECP_CONST uint64_t SECP_NC[4] = {
    0x402DA1732FC9BEBFULL,
    0x4551231950B75FC4ULL,
    0x1ULL,
    0x0ULL,
};

/* Reduce (carry * 2^256 + k) modulo n, storing the result in [0, n) in k.
 * carry must be 0 or 1. */
SECP_INLINE void secp_mod_n(uint64_t *k, uint64_t carry) {
    while (carry) {
        unsigned __int128 c = (unsigned __int128)k[0] + SECP_NC[0];
        k[0] = (uint64_t)c; c >>= 64;
        c = (unsigned __int128)k[1] + SECP_NC[1] + (uint64_t)c;
        k[1] = (uint64_t)c; c >>= 64;
        c = (unsigned __int128)k[2] + SECP_NC[2] + (uint64_t)c;
        k[2] = (uint64_t)c; c >>= 64;
        c = (unsigned __int128)k[3] + SECP_NC[3] + (uint64_t)c;
        k[3] = (uint64_t)c; c >>= 64;
        carry = (uint64_t)c;
    }
    /* At most one conditional subtraction of n (k < 2^256 < 2n). */
    uint64_t q[4];
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 s = (unsigned __int128)k[i] - SECP_N[i] - (uint64_t)borrow;
        q[i] = (uint64_t)s;
        borrow = (s >> 64) & 1ULL;
    }
    if (borrow == 0) {
        for (int i = 0; i < 4; i++) k[i] = q[i];
    }
}

/* R = P + Q in affine coordinates, given inv = 1/(Qx - Px) mod p.
 * P and Q must be distinct and not inverses of each other (Qx != Px).
 * Aliasing-safe: outputs may alias inputs. */
SECP_INLINE void secp_affine_add_inv(
    uint64_t *rx, uint64_t *ry,
    const uint64_t *px, const uint64_t *py,
    const uint64_t *qx, const uint64_t *qy,
    const uint64_t *inv) {
    uint64_t PX[4], PY[4], QX[4], QY[4], IV[4];
    for (int i = 0; i < 4; i++) { PX[i] = px[i]; PY[i] = py[i]; QX[i] = qx[i]; QY[i] = qy[i]; IV[i] = inv[i]; }
    uint64_t s[4], t[4];
    secp_sub(t, QY, PY);     /* Qy - Py */
    secp_mul(s, t, IV);      /* lambda = (Qy-Py) / (Qx-Px) */
    secp_sqr(t, s);          /* lambda^2 */
    secp_sub(t, t, PX);      /* - Px */
    secp_sub(rx, t, QX);     /* Rx = lambda^2 - Px - Qx */
    secp_sub(t, PX, rx);     /* Px - Rx */
    secp_mul(t, s, t);       /* lambda*(Px-Rx) */
    secp_sub(ry, t, PY);     /* Ry = lambda*(Px-Rx) - Py */
}

/* Batch-invert z[0..count-1] in place.  prefix must hold count*4 limbs.
 * Every z_i must be non-zero. */
SECP_INLINE void secp_batch_invert(uint64_t *z, uint64_t *prefix, int count) {
    if (count <= 0) return;
    for (int w = 0; w < 4; w++) prefix[w] = z[w];
    for (int i = 1; i < count; i++)
        secp_mul(prefix + 4 * i, prefix + 4 * (i - 1), z + 4 * i);
    uint64_t inv[4];
    secp_inv(inv, prefix + 4 * (count - 1));
    uint64_t u[4] = {inv[0], inv[1], inv[2], inv[3]};
    for (int i = count; i-- > 0;) {
        if (i == 0) {
            for (int w = 0; w < 4; w++) z[w] = u[w];
        } else {
            uint64_t zi[4];
            secp_mul(zi, u, prefix + 4 * (i - 1));
            secp_mul(u, u, z + 4 * i); /* original z[i], still intact */
            for (int w = 0; w < 4; w++) z[4 * i + w] = zi[w];
        }
    }
}

/* Write a 4-limb (little-endian) field element as 32 big-endian bytes, via
 * four byte-swapped 64-bit stores.  `byte_off` must be 8-byte aligned (the
 * kernel always writes at 64-byte key boundaries).  Eight wide stores per key
 * instead of 64 byte stores keeps the scattered per-thread writes from
 * saturating the store pipeline. */
SECP_INLINE void secp_write_be32(uint8_t *out, uint64_t byte_off, const uint64_t *limbs) {
    uint64_t *dst = (uint64_t *)(out + byte_off);
    for (int w = 0; w < 4; w++) {
        uint64_t v = limbs[w];
        /* bswap64: reverse the byte order so little-endian limb w lands at
         * big-endian word position 3 - w. */
        v = ((v & 0x00FF00FF00FF00FFULL) << 8) | ((v >> 8) & 0x00FF00FF00FF00FFULL);
        v = ((v & 0x0000FFFF0000FFFFULL) << 16) | ((v >> 16) & 0x0000FFFF0000FFFFULL);
        v = (v << 32) | (v >> 32);
        dst[3 - w] = v;
    }
}

#endif /* SECP256K1_GROUPED_H */
