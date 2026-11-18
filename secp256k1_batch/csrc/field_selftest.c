/* Standalone self-test for secp256k1_field.h. Build with:
 *   cc -O2 -std=c11 -o /tmp/field_test field_selftest.c
 * Reads pairs of hex limbs from argv and prints results; the Python driver
 * cross-checks against gmpy2. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "secp256k1_field.h"

static uint64_t parse(const char *s) { return strtoull(s, NULL, 0); }

static void print4(const uint64_t *x) {
    printf("%016llx%016llx%016llx%016llx",
           (unsigned long long)x[3], (unsigned long long)x[2],
           (unsigned long long)x[1], (unsigned long long)x[0]);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    /* argv[1..8]: a (4 limbs, limb3..limb0 hex words? use full 16-hex words) */
    /* Simplify: read 8 x 64-bit hex words for a (4) and b (4), little-endian order. */
    uint64_t a[4], b[4], r[4];
    for (int i = 0; i < 4; i++) a[i] = parse(argv[1 + i]);
    for (int i = 0; i < 4; i++) b[i] = parse(argv[5 + i]);
    secp_mul(r, a, b);
    printf("mul ");
    print4(r);
    printf("\n");
    secp_add(r, a, b);
    printf("add ");
    print4(r);
    printf("\n");
    secp_sub(r, a, b);
    printf("sub ");
    print4(r);
    printf("\n");
    secp_sqr(r, a);
    printf("sqr ");
    print4(r);
    printf("\n");
    secp_inv(r, a);
    printf("inv ");
    print4(r);
    printf("\n");
    return 0;
}
