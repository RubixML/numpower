#ifndef PHPSCI_NDARRAY_DD_MATH_H
#define PHPSCI_NDARRAY_DD_MATH_H

/* Double-double arithmetic primitives.
   Used as the storage + arithmetic backend for the float128 dtype on
   platforms where __float128 / libquadmath is not available (Apple clang
   on macOS, MSVC on Windows). Provides ~106 effective mantissa bits
   (~32 decimal digits) — the same precision tier the project already
   uses for fp128 on GPU.

   On Linux GCC x86/x64 the build keeps native __float128 + libquadmath,
   and this header's symbols stay unused. */

#include <stddef.h>

typedef struct {
    double hi;
    double lo;
} ndarray_dd_t;

/* ── constructors / accessors ───────────────────────────────────────────── */
ndarray_dd_t ndarray_dd_from_double(double a);
ndarray_dd_t ndarray_dd_from_pair(double hi, double lo);
double       ndarray_dd_to_double(ndarray_dd_t a);
long long    ndarray_dd_to_int64(ndarray_dd_t a);
ndarray_dd_t ndarray_dd_from_int64(long long a);

/* ── arithmetic ─────────────────────────────────────────────────────────── */
ndarray_dd_t ndarray_dd_neg(ndarray_dd_t a);
ndarray_dd_t ndarray_dd_abs(ndarray_dd_t a);
ndarray_dd_t ndarray_dd_add(ndarray_dd_t a, ndarray_dd_t b);
ndarray_dd_t ndarray_dd_sub(ndarray_dd_t a, ndarray_dd_t b);
ndarray_dd_t ndarray_dd_mul(ndarray_dd_t a, ndarray_dd_t b);
ndarray_dd_t ndarray_dd_div(ndarray_dd_t a, ndarray_dd_t b);
ndarray_dd_t ndarray_dd_trunc(ndarray_dd_t a);
ndarray_dd_t ndarray_dd_fmod(ndarray_dd_t a, ndarray_dd_t b);
ndarray_dd_t ndarray_dd_pow(ndarray_dd_t a, ndarray_dd_t b);

/* Square root and reciprocal sqrt to ~106-bit precision via one
   double-precision sqrt seed followed by a single Newton iteration in
   double-double arithmetic. Returns NaN for negative inputs and matches
   sqrtq(__float128)/qsqrtq on the libquadmath build to the last DD bit. */
ndarray_dd_t ndarray_dd_sqrt(ndarray_dd_t a);
ndarray_dd_t ndarray_dd_rsqrt(ndarray_dd_t a);

/* ── comparisons ────────────────────────────────────────────────────────── */
int  ndarray_dd_cmp(ndarray_dd_t a, ndarray_dd_t b);   /* -1, 0, 1 */
int  ndarray_dd_iszero(ndarray_dd_t a);
int  ndarray_dd_isnan(ndarray_dd_t a);
int  ndarray_dd_isinf(ndarray_dd_t a);

/* ── string I/O ─────────────────────────────────────────────────────────── */
/* Parses a decimal string into a DD value with up to ~32 significant
   digits of precision. Accepts the same forms strtod accepts plus
   "inf"/"-inf"/"nan" (case-insensitive). */
ndarray_dd_t ndarray_dd_from_string(const char *str);

/* Formats `a` as a decimal string into `buf` (max `bufsize` bytes including
   the NUL terminator). Uses scientific notation when the magnitude is far
   from 1; otherwise plain decimal. Outputs up to 32 significant digits and
   trims trailing zeros. */
void ndarray_dd_to_string(ndarray_dd_t a, char *buf, size_t bufsize);

#endif /* PHPSCI_NDARRAY_DD_MATH_H */
