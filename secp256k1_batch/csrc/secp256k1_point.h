/* secp256k1 group arithmetic in Jacobian coordinates, shared by the CPU and
 * CUDA backends.
 *
 * A point is (X, Y, Z) with affine coordinates (X/Z^2, Y/Z^3). The point at
 * infinity is represented by Z == 0. All field elements are canonical
 * (in [0, p)) at function boundaries.
 *
 * All point functions read their inputs into locals before writing outputs, so
 * they are safe to call with aliased input/output pointers (r == p), which the
 * scalar-multiplication and increment loops rely on.
 *
 * The hot path is a mixed addition (Jacobian + affine generator), which avoids
 * a field inversion per step -- that is the single most important optimization
 * for generating a sequential range of public keys.
 */
#ifndef SECP256K1_POINT_H
#define SECP256K1_POINT_H

#include "secp256k1_field.h"

typedef struct {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
} secp_pt;

/* The generator G in little-endian 64-bit limbs (matching constants.py). */
SECP_CONST uint64_t SECP_GX[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL,
};
SECP_CONST uint64_t SECP_GY[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL,
};

/* Infinity: z == 0. */
SECP_INLINE int secp_pt_is_inf(const secp_pt *p) { return secp_is_zero(p->z); }
SECP_INLINE void secp_pt_set_inf(secp_pt *p) {
    p->z[0] = p->z[1] = p->z[2] = p->z[3] = 0;
}

/* r = p + p  (Jacobian doubling, dbl-2009-l specialized for a = 0).
 * Aliasing-safe: r may equal p. */
SECP_INLINE void secp_pt_double(secp_pt *r, const secp_pt *p) {
    uint64_t X1[4], Y1[4], Z1[4];
    for (int i = 0; i < 4; i++) { X1[i] = p->x[i]; Y1[i] = p->y[i]; Z1[i] = p->z[i]; }

    if (secp_is_zero(Z1) || secp_is_zero(Y1)) {
        secp_pt_set_inf(r);
        return;
    }
    uint64_t A[4], B[4], C[4], D[4], E[4], F[4], t[4];
    secp_sqr(A, X1);          /* A = X^2 */
    secp_sqr(B, Y1);          /* B = Y^2 */
    secp_sqr(C, B);           /* C = Y^4 */
    secp_add(t, X1, B);       /* X + B */
    secp_sqr(t, t);           /* (X + B)^2 */
    secp_sub(t, t, A);        /* - A */
    secp_sub(t, t, C);        /* - C */
    secp_add(D, t, t);        /* D = 2*((X+B)^2 - A - C) */
    secp_add(E, A, A);        /* 2A */
    secp_add(E, E, A);        /* E = 3A */
    secp_sqr(F, E);           /* F = E^2 */
    secp_add(t, D, D);        /* 2D */
    secp_sub(r->x, F, t);     /* X3 = F - 2D */
    secp_sub(t, D, r->x);     /* D - X3 */
    secp_mul(t, E, t);        /* E*(D - X3) */
    secp_add(r->y, C, C);     /* 2C */
    secp_add(r->y, r->y, r->y); /* 4C */
    secp_add(r->y, r->y, r->y); /* 8C */
    secp_sub(r->y, t, r->y);  /* Y3 = E*(D-X3) - 8C */
    secp_mul(r->z, Y1, Z1);   /* Y*Z */
    secp_add(r->z, r->z, r->z); /* Z3 = 2*Y*Z */
}

/* r = p + Q, where p is Jacobian and Q = (qx, qy) is affine.
 * Mixed addition (madd-2004-hmv). Aliasing-safe: r may equal p. */
SECP_INLINE void secp_pt_add_affine(secp_pt *r, const secp_pt *p,
                                    const uint64_t *qx, const uint64_t *qy) {
    uint64_t X1[4], Y1[4], Z1[4];
    for (int i = 0; i < 4; i++) { X1[i] = p->x[i]; Y1[i] = p->y[i]; Z1[i] = p->z[i]; }
    uint64_t gx[4], gy[4];
    for (int i = 0; i < 4; i++) { gx[i] = qx[i]; gy[i] = qy[i]; }

    if (secp_is_zero(Z1)) {
        for (int i = 0; i < 4; i++) { r->x[i] = gx[i]; r->y[i] = gy[i]; }
        r->z[0] = 1; r->z[1] = r->z[2] = r->z[3] = 0;
        return;
    }
    uint64_t Z1Z1[4], U2[4], S2[4], H[4], r2[4], HH[4], I[4], J[4], V[4], t[4];

    secp_sqr(Z1Z1, Z1);       /* Z1Z1 = Z1^2 */
    secp_mul(U2, gx, Z1Z1);   /* U2 = Gx * Z1Z1 */
    secp_mul(t, Z1, Z1Z1);    /* Z1 * Z1Z1 = Z1^3 */
    secp_mul(S2, gy, t);      /* S2 = Gy * Z1^3 */

    secp_sub(H, U2, X1);      /* H = U2 - X1 */
    secp_sub(r2, S2, Y1);     /* S2 - Y1 */

    if (secp_is_zero(H)) {
        if (secp_is_zero(r2)) {
            secp_pt tmp;
            for (int i = 0; i < 4; i++) { tmp.x[i] = X1[i]; tmp.y[i] = Y1[i]; tmp.z[i] = Z1[i]; }
            secp_pt_double(r, &tmp); /* P == Q */
        } else {
            secp_pt_set_inf(r);       /* P == -Q */
        }
        return;
    }

    secp_sqr(HH, H);          /* HH = H^2 */
    secp_add(I, HH, HH);      /* 2HH */
    secp_add(I, I, I);        /* I = 4HH */
    secp_mul(J, H, I);        /* J = H*I = 4 H^3 */
    secp_add(t, r2, r2);      /* r = 2*(S2 - Y1) */
    secp_mul(V, X1, I);       /* V = X1 * I */

    secp_sqr(r2, t);          /* r^2 */
    secp_add(t, V, V);        /* 2V */
    secp_sub(r2, r2, t);      /* r^2 - 2V */
    secp_sub(r->x, r2, J);    /* X3 = r^2 - J - 2V */

    /* Y3 = r*(V - X3) - 2*Y1*J, where r = 2*(S2 - Y1). */
    {
        uint64_t r_val[4];
        secp_sub(r_val, S2, Y1);     /* S2 - Y1 */
        secp_add(r_val, r_val, r_val); /* r = 2*(S2 - Y1) */
        secp_sub(t, V, r->x);        /* V - X3 */
        secp_mul(r->y, r_val, t);    /* r*(V - X3) */
        secp_mul(t, Y1, J);          /* Y1*J */
        secp_add(t, t, t);           /* 2*Y1*J */
        secp_sub(r->y, r->y, t);     /* Y3 = r*(V-X3) - 2*Y1*J */
    }

    /* Z3 = (Z1 + H)^2 - Z1Z1 - HH */
    secp_add(t, Z1, H);       /* Z1 + H */
    secp_sqr(t, t);           /* (Z1+H)^2 */
    secp_sub(t, t, Z1Z1);     /* - Z1Z1 */
    secp_sub(r->z, t, HH);    /* Z3 = (Z1+H)^2 - Z1Z1 - HH */
}

/* r = p + G, the affine generator (thin wrapper around secp_pt_add_affine that
 * reads G from __constant__ memory). */
SECP_INLINE void secp_pt_add_generator(secp_pt *r, const secp_pt *p) {
    uint64_t gx[4], gy[4];
    for (int i = 0; i < 4; i++) { gx[i] = SECP_GX[i]; gy[i] = SECP_GY[i]; }
    secp_pt_add_affine(r, p, gx, gy);
}

/* r = k * G. Double-and-add over the 256-bit scalar. */
SECP_INLINE void secp_scalar_mult(secp_pt *r, const uint64_t *k) {
    secp_pt_set_inf(r);
    int started = 0;
    for (int bit = 255; bit >= 0; bit--) {
        if (started) secp_pt_double(r, r);
        int limb = bit / 64;
        int off = bit % 64;
        if ((k[limb] >> off) & 1ULL) {
            secp_pt_add_generator(r, r);
            started = 1;
        }
    }
}

/* Convert a Jacobian point to affine (rx, ry). Requires z != 0. */
SECP_INLINE void secp_pt_to_affine(uint64_t *rx, uint64_t *ry, const secp_pt *p) {
    uint64_t zinv[4], zinv2[4], zinv3[4];
    secp_inv(zinv, p->z);
    secp_sqr(zinv2, zinv);
    secp_mul(zinv3, zinv2, zinv);
    secp_mul(rx, p->x, zinv2);
    secp_mul(ry, p->y, zinv3);
}

#endif /* SECP256K1_POINT_H */
