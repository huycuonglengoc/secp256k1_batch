/* secp256k1 256-bit field arithmetic (mod p), shared by the CPU and CUDA
 * backends.
 *
 * A field element is four little-endian 64-bit limbs.  The prime is
 *
 *     p = 2^256 - 2^32 - 977
 *
 * which gives the reduction identity  2^256 == C (mod p)  with C = 2^32 + 977.
 *
 * Multiplication produces a 512-bit product (8 limbs); the product is reduced
 * to a canonical element of [0, p) using two high-limb folds followed by a
 * conditional subtraction.  The code only needs 64x64 -> 128-bit products, so
 * it maps cleanly onto `unsigned __int128` (GCC/Clang) and the CUDA `long
 * long` extension used by nvcc.
 *
 * The same header is #included by the CPU shared library and the CUDA kernel,
 * so a single implementation is validated by both backends.
 */
#ifndef SECP256K1_FIELD_H
#define SECP256K1_FIELD_H

/* NVRTC (CuPy's default runtime compiler) does not ship <stdint.h>; nvcc and
 * the host compilers do. Provide the exact-width types ourselves under NVRTC. */
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
#else
#include <stdint.h>
#endif

#if defined(__CUDACC__)
#define SECP_INLINE __device__ static inline
#define SECP_CONST __constant__
#else
#define SECP_INLINE static inline
#define SECP_CONST static const
#endif

/* p as four little-endian limbs. */
SECP_CONST uint64_t SECP_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
};

/* C = 2^32 + 977  (the value congruent to 2^256 mod p). */
static const uint64_t SECP_C = 0x1000003D1ULL;

#define SECP_MASK64 0xFFFFFFFFFFFFFFFFULL

/* ---- Basic limb helpers -------------------------------------------------- */

/* Add the n-limb value b into a (in place). A final carry beyond limb n-1 is
 * dropped; callers guarantee that cannot happen for their inputs. */
SECP_INLINE void secp_add_limbs(uint64_t *a, const uint64_t *b, int n) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < n; i++) {
        unsigned __int128 s = (unsigned __int128)a[i] + b[i] + (uint64_t)carry;
        a[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

/* r = (a + b) mod p, for a, b already in [0, p). */
SECP_INLINE void secp_add(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    unsigned __int128 carry = 0;
    uint64_t t[4];
    for (int i = 0; i < 4; i++) {
        unsigned __int128 s = (unsigned __int128)a[i] + b[i] + (uint64_t)carry;
        t[i] = (uint64_t)s;
        carry = s >> 64;
    }
    /* t may be up to 2^257 - 2. If carry (bit 256) is set, or t >= p, subtract p.
     * We fold a set carry as one subtraction of p (since 2^256 == C < p is not
     * usable directly here; subtracting p from t+2^256 is correct). */
    if (carry) {
        unsigned __int128 borrow = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 s = (unsigned __int128)t[i] - SECP_P[i] - (uint64_t)borrow;
            r[i] = (uint64_t)s;
            borrow = (s >> 64) & 1ULL;
        }
        return;
    }
    /* No carry: subtract p if t >= p. */
    {
        unsigned __int128 borrow = 0;
        uint64_t q[4];
        for (int i = 0; i < 4; i++) {
            unsigned __int128 s = (unsigned __int128)t[i] - SECP_P[i] - (uint64_t)borrow;
            q[i] = (uint64_t)s;
            borrow = (s >> 64) & 1ULL;
        }
        if (borrow == 0) { /* t >= p */
            for (int i = 0; i < 4; i++) r[i] = q[i];
        } else {
            for (int i = 0; i < 4; i++) r[i] = t[i];
        }
    }
}

/* r = (a - b) mod p, for a, b in [0, p). */
SECP_INLINE void secp_sub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    unsigned __int128 borrow = 0;
    uint64_t t[4];
    for (int i = 0; i < 4; i++) {
        unsigned __int128 s = (unsigned __int128)a[i] - b[i] - (uint64_t)borrow;
        t[i] = (uint64_t)s;
        borrow = (s >> 64) & 1ULL;
    }
    if (borrow) { /* a < b: add p back */
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 s = (unsigned __int128)t[i] + SECP_P[i] + (uint64_t)carry;
            r[i] = (uint64_t)s;
            carry = s >> 64;
        }
    } else {
        for (int i = 0; i < 4; i++) r[i] = t[i];
    }
}

/* Is a equal to b? Constant-shape comparison (used for the infinity test). */
SECP_INLINE int secp_equal(const uint64_t *a, const uint64_t *b) {
    uint64_t diff = 0;
    for (int i = 0; i < 4; i++) diff |= a[i] ^ b[i];
    return diff == 0;
}

/* Is a zero? */
SECP_INLINE int secp_is_zero(const uint64_t *a) {
    uint64_t acc = 0;
    for (int i = 0; i < 4; i++) acc |= a[i];
    return acc == 0;
}

/* ---- Multiplication and reduction --------------------------------------- */

#if defined(__CUDA_ARCH__)
/* PTX inline assembly for the 64x64 -> 128-bit multiply-accumulate hot path.
 * This mirrors KeyHunt / VanitySearch's GPUMath.h: native mul.lo/mul.hi and
 * carry-chained add.cc/addc are tighter than emulated `__int128`.  Only used
 * in device code; the host keeps the portable `__int128` version below. */

#define SECP_MUL_LO(c, a, b) asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(c) : "l"((uint64_t)(a)), "l"((uint64_t)(b)))
#define SECP_MUL_HI(c, a, b) asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(c) : "l"((uint64_t)(a)), "l"((uint64_t)(b)))
#define SECP_ADD_CC(c, b) asm volatile("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"((uint64_t)(b)))
#define SECP_ADDC(c, b) asm volatile("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"((uint64_t)(b)))

/* (t, carry) += ai * bj, i.e. t = low64(ai*bj + t + carry), carry = high64. */
SECP_INLINE void secp_mac(uint64_t &t, uint64_t &carry, uint64_t ai, uint64_t bj) {
    uint64_t lo, hi;
    SECP_MUL_LO(lo, ai, bj);
    SECP_MUL_HI(hi, ai, bj);
    SECP_ADD_CC(lo, t);     /* lo += t            (CC = c1) */
    SECP_ADDC(hi, 0);       /* hi += c1 */
    SECP_ADD_CC(lo, carry); /* lo += carry        (CC = c2) */
    SECP_ADDC(hi, 0);       /* hi += c2 */
    t = lo;
    carry = hi;
}

/* t = a * b as an 8-limb (512-bit) little-endian product (row method). */
SECP_INLINE void secp_mul512(uint64_t t[8], const uint64_t *a, const uint64_t *b) {
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    uint64_t carry;

    carry = 0;
    secp_mac(t0, carry, a0, b0);
    secp_mac(t1, carry, a0, b1);
    secp_mac(t2, carry, a0, b2);
    secp_mac(t3, carry, a0, b3);
    t4 = carry;

    carry = 0;
    secp_mac(t1, carry, a1, b0);
    secp_mac(t2, carry, a1, b1);
    secp_mac(t3, carry, a1, b2);
    secp_mac(t4, carry, a1, b3);
    t5 = carry;

    carry = 0;
    secp_mac(t2, carry, a2, b0);
    secp_mac(t3, carry, a2, b1);
    secp_mac(t4, carry, a2, b2);
    secp_mac(t5, carry, a2, b3);
    t6 = carry;

    carry = 0;
    secp_mac(t3, carry, a3, b0);
    secp_mac(t4, carry, a3, b1);
    secp_mac(t5, carry, a3, b2);
    secp_mac(t6, carry, a3, b3);
    t7 = carry;

    t[0] = t0; t[1] = t1; t[2] = t2; t[3] = t3;
    t[4] = t4; t[5] = t5; t[6] = t6; t[7] = t7;
}

#else

/* t = a * b as an 8-limb (512-bit) little-endian product. */
SECP_INLINE void secp_mul512(uint64_t t[8], const uint64_t *a, const uint64_t *b) {
    for (int i = 0; i < 8; i++) t[i] = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 cur = (unsigned __int128)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
        t[i + 4] = carry;
    }
}

#endif /* __CUDA_ARCH__ */

/* Fold a 512-bit product into a 5-limb value < 2^289 using 2^256 == C.
 *   u = low(t) + (high(t) << 32) + high(t) * 977   (mod p)
 * where high(t) = t[4..7]. The result has u[4] < 2^33. */
SECP_INLINE void secp_fold512(uint64_t u[5], const uint64_t *t) {
    const uint64_t *h = t + 4;
    for (int i = 0; i < 4; i++) u[i] = t[i];
    u[4] = 0;

    /* + (h << 32) */
    uint64_t s[5];
    s[0] = h[0] << 32;
    s[1] = (h[1] << 32) | (h[0] >> 32);
    s[2] = (h[2] << 32) | (h[1] >> 32);
    s[3] = (h[3] << 32) | (h[2] >> 32);
    s[4] = h[3] >> 32;
    secp_add_limbs(u, s, 5);

    /* + (h * 977) */
    uint64_t m[5] = {0, 0, 0, 0, 0};
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 prod = (unsigned __int128)h[i] * 977ULL + (uint64_t)carry;
        m[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    m[4] = (uint64_t)carry;
    secp_add_limbs(u, m, 5);
}

/* Reduce a 5-limb value (u[4] < 2^64) to a canonical 4-limb element of [0,p).
 * Works by repeatedly folding the top limb (u[4] * 2^256 == u[4] * C) until the
 * value fits in 256 bits, then one conditional subtraction of p. */
SECP_INLINE void secp_reduce5(uint64_t u[5]) {
    while (u[4]) {
        unsigned __int128 carry = (unsigned __int128)u[4] * SECP_C;
        u[4] = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 s = (unsigned __int128)u[i] + (uint64_t)(carry & SECP_MASK64);
            u[i] = (uint64_t)s;
            carry = (carry >> 64) + (s >> 64);
        }
        u[4] = (uint64_t)carry;
    }
    /* u[4] == 0, value < 2^256. Subtract p once if value >= p. */
    {
        uint64_t q[4];
        unsigned __int128 borrow = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 s = (unsigned __int128)u[i] - SECP_P[i] - (uint64_t)borrow;
            q[i] = (uint64_t)s;
            borrow = (s >> 64) & 1ULL;
        }
        if (borrow == 0) {
            for (int i = 0; i < 4; i++) u[i] = q[i];
        }
    }
}

/* r = (a * b) mod p. */
SECP_INLINE void secp_mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t t[8];
    secp_mul512(t, a, b);
    uint64_t u[5];
    secp_fold512(u, t);
    secp_reduce5(u);
    for (int i = 0; i < 4; i++) r[i] = u[i];
}

/* Add the 128-bit value (hi << 64 | lo) into the 8-limb accumulator t starting
 * at limb k, propagating carries upward. */
SECP_INLINE void secp_add128_at(uint64_t t[8], uint64_t lo, uint64_t hi, int k) {
    unsigned __int128 c = (unsigned __int128)t[k] + lo;
    t[k] = (uint64_t)c;
    c = (unsigned __int128)t[k + 1] + hi + (uint64_t)(c >> 64);
    t[k + 1] = (uint64_t)c;
    c >>= 64;
    for (int j = k + 2; j < 8 && c; j++) {
        c = (unsigned __int128)t[j] + (uint64_t)c;
        t[j] = (uint64_t)c;
        c >>= 64;
    }
}

/* r = a^2 mod p, exploiting the a_i*a_j == a_j*a_i symmetry (10 products
 * instead of 16). */
SECP_INLINE void secp_sqr(uint64_t *r, const uint64_t *a) {
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        unsigned __int128 d = (unsigned __int128)a[i] * a[i];
        secp_add128_at(t, (uint64_t)d, (uint64_t)(d >> 64), 2 * i);
        for (int j = i + 1; j < 4; j++) {
            unsigned __int128 p = (unsigned __int128)a[i] * a[j];
            secp_add128_at(t, (uint64_t)p, (uint64_t)(p >> 64), i + j);
            secp_add128_at(t, (uint64_t)p, (uint64_t)(p >> 64), i + j);
        }
    }
    uint64_t u[5];
    secp_fold512(u, t);
    secp_reduce5(u);
    for (int i = 0; i < 4; i++) r[i] = u[i];
}

/* r = a^(-1) mod p, via the addition chain for p - 2 = 2^256 - 2^32 - 979
 * (Brian Smith's chain, as used by libsecp256k1).  About 267 squarings and
 * 15 multiplications instead of the ~510 field operations of naive Fermat. */
SECP_INLINE void secp_inv(uint64_t *r, const uint64_t *a) {
    uint64_t x2[4], x3[4], x63[4], x511[4], x2047[4];
    uint64_t x22[4], x44[4], x88[4], x176[4], x220[4], x223[4];
    uint64_t x14[4], x15[4], x45[4], B[4], C[4], t[4];
    int i;

    /* x2 = x^3 */
    secp_sqr(x2, a);
    secp_mul(x2, x2, a);
    /* x3 = x^7 */
    secp_sqr(x3, x2);
    secp_mul(x3, x3, a);

    /* x63 = x^(2^6 - 1) */
    for (i = 0; i < 4; i++) x63[i] = x3[i];
    for (i = 0; i < 3; i++) secp_sqr(x63, x63);
    secp_mul(x63, x63, x3);
    /* x511 = x^(2^9 - 1) */
    for (i = 0; i < 4; i++) x511[i] = x63[i];
    for (i = 0; i < 3; i++) secp_sqr(x511, x511);
    secp_mul(x511, x511, x3);
    /* x2047 = x^(2^11 - 1) */
    for (i = 0; i < 4; i++) x2047[i] = x511[i];
    for (i = 0; i < 2; i++) secp_sqr(x2047, x2047);
    secp_mul(x2047, x2047, x2);
    /* x22 = x^(2^22 - 1) */
    for (i = 0; i < 4; i++) x22[i] = x2047[i];
    for (i = 0; i < 11; i++) secp_sqr(x22, x22);
    secp_mul(x22, x22, x2047);
    /* x44 = x^(2^44 - 1) */
    for (i = 0; i < 4; i++) x44[i] = x22[i];
    for (i = 0; i < 22; i++) secp_sqr(x44, x44);
    secp_mul(x44, x44, x22);
    /* x88 = x^(2^88 - 1) */
    for (i = 0; i < 4; i++) x88[i] = x44[i];
    for (i = 0; i < 44; i++) secp_sqr(x88, x88);
    secp_mul(x88, x88, x44);
    /* x176 = x^(2^176 - 1) */
    for (i = 0; i < 4; i++) x176[i] = x88[i];
    for (i = 0; i < 88; i++) secp_sqr(x176, x176);
    secp_mul(x176, x176, x88);
    /* x220 = x^(2^220 - 1) */
    for (i = 0; i < 4; i++) x220[i] = x176[i];
    for (i = 0; i < 44; i++) secp_sqr(x220, x220);
    secp_mul(x220, x220, x44);
    /* x223 = x^(2^223 - 1) */
    for (i = 0; i < 4; i++) x223[i] = x220[i];
    for (i = 0; i < 3; i++) secp_sqr(x223, x223);
    secp_mul(x223, x223, x3);

    /* B = x^(2^32 - 979) = x^(2^32 - 2^10) * x^45, with
     *   x^45 = x^(32+8+4+1) = ((x^7^2 * x)^2) * (x^7^2 * x)  */
    secp_sqr(x14, x3);        /* x^14 */
    secp_mul(x15, x14, a);    /* x^15 */
    secp_sqr(t, x15);         /* x^30 */
    secp_mul(x45, t, x15);    /* x^45 */
    for (i = 0; i < 4; i++) B[i] = x22[i];
    for (i = 0; i < 10; i++) secp_sqr(B, B);   /* x^(2^32 - 2^10) */
    secp_mul(B, B, x45);                        /* x^(2^32 - 979) */

    /* C = x^(2^256 - 2^33) = (x^(2^223 - 1))^(2^33) */
    for (i = 0; i < 4; i++) C[i] = x223[i];
    for (i = 0; i < 33; i++) secp_sqr(C, C);

    /* r = C * B = x^(2^256 - 2^32 - 979) = x^(p-2) */
    secp_mul(r, C, B);
}

/* r = -a mod p. */
SECP_INLINE void secp_neg(uint64_t *r, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    secp_sub(r, zero, a);
}

#endif /* SECP256K1_FIELD_H */
