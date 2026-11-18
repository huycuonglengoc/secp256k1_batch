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

/* r = a^2 mod p. */
SECP_INLINE void secp_sqr(uint64_t *r, const uint64_t *a) {
    secp_mul(r, a, a);
}

/* r = a^(-1) mod p via Fermat (a^(p-2)). ~256 squarings + multiplications. */
SECP_INLINE void secp_inv(uint64_t *r, const uint64_t *a) {
    /* p - 2 = 2^256 - 2^32 - 979. Binary expansion: all 256 bits set except a
     * few low ones. We compute with a left-to-right square-and-multiply on the
     * full 256-bit exponent 0xFFFFFFFEFFFFFC2F - ... ; simpler: use the literal
     * exponent p - 2 whose binary form we generate here. */
    uint64_t e[4] = {
        0xFFFFFFFEFFFFFC2DULL, /* p - 2 low limb */
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
    };
    uint64_t res[4] = {1, 0, 0, 0};
    uint64_t base[4] = {a[0], a[1], a[2], a[3]};
    int started = 0;
    for (int bit = 255; bit >= 0; bit--) {
        if (started) {
            secp_sqr(res, res);
        }
        int limb = bit / 64;
        int off = bit % 64;
        if ((e[limb] >> off) & 1ULL) {
            if (started) {
                secp_mul(res, res, base);
            } else {
                /* first set bit: res = base */
                for (int i = 0; i < 4; i++) res[i] = base[i];
                started = 1;
            }
        }
    }
    for (int i = 0; i < 4; i++) r[i] = res[i];
}

/* r = -a mod p. */
SECP_INLINE void secp_neg(uint64_t *r, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    secp_sub(r, zero, a);
}

#endif /* SECP256K1_FIELD_H */
