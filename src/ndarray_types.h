#ifndef PHPSCI_NDARRAY_TYPES_EXTRA_H
#define PHPSCI_NDARRAY_TYPES_EXTRA_H

#include <stdint.h>
#include <stddef.h>

/* ── float4 : 4-bit float, E2M1 format, stored in lower nibble of uint8_t ─ */
double  ndarray_fp4_to_double(uint8_t nibble);
uint8_t ndarray_double_to_fp4(double val);

/* ── float8 : E4M3 8-bit float ─────────────────────────────────────────── */
double  ndarray_fp8_to_double(uint8_t fp8);
uint8_t ndarray_double_to_fp8(double val);

/* ── float16 : IEEE-754 half precision ──────────────────────────────────── */
double   ndarray_fp16_to_double(uint16_t fp16);
uint16_t ndarray_double_to_fp16(double val);

/* ── float128 ───────────────────────────────────────────────────────────── */
/* On Linux GCC x86 we get full 113-bit IEEE 754 binary128 via __float128 +
   libquadmath (when present). On every other platform we fall back to a
   double-double struct (~106 mantissa bits, ~32 decimal digits) — same
   precision tier the project already uses for fp128 on GPU. Storage is
   always 16 bytes, independent of host long-double width, so the layout
   round-trips through serialize() and GPU/CPU transfers identically. */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__APPLE__) && \
    (defined(__x86_64__) || defined(__i386__))
#  define NDARRAY_HAVE_FLOAT128 1
   typedef __float128 ndarray_fp128_t;
#  define NDARRAY_FP128_SIZE 16
#else
#  define NDARRAY_HAVE_FLOAT128 0
#  define NDARRAY_FP128_USES_DD 1
#  include "dd_math.h"
   typedef ndarray_dd_t ndarray_fp128_t;
#  define NDARRAY_FP128_SIZE 16
#endif

/* ── platform-agnostic ops ──────────────────────────────────────────────── */
/* Sites that do fp128 arithmetic use these macros instead of bare operators
   so the call code is identical on both backends.  On the __float128 path
   they expand to native operators (zero overhead); on the DD path they
   expand to ndarray_dd_* calls. */
#if NDARRAY_HAVE_FLOAT128
   /* sqrt / sin / isnan for __float128 are routed through these out-of-line
      helpers so the libquadmath-vs-libm-fallback choice is made inside
      ndarray_types.c (where `config.h` has been included and `HAVE_QUADMATH`
      is meaningful) rather than baked into the expansion at every header
      include site. Inlining the choice here would silently pick the
      long-double fallback in any translation unit that pulls
      ndarray_types.h via a transitive header before its own
      `#include "config.h"`. */
   ndarray_fp128_t ndarray_fp128_sqrt(ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_sin (ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_exp  (ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_exp2 (ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_expm1(ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_log  (ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_log1p(ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_log2 (ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_log10(ndarray_fp128_t a);
   ndarray_fp128_t ndarray_fp128_logb (ndarray_fp128_t a);
   int             ndarray_fp128_isnan(ndarray_fp128_t a);
#  define NDARRAY_FP128_SQRT(a)     ndarray_fp128_sqrt(a)
#  define NDARRAY_FP128_SIN(a)      ndarray_fp128_sin(a)
#  define NDARRAY_FP128_EXP(a)      ndarray_fp128_exp(a)
#  define NDARRAY_FP128_EXP2(a)     ndarray_fp128_exp2(a)
#  define NDARRAY_FP128_EXPM1(a)    ndarray_fp128_expm1(a)
#  define NDARRAY_FP128_LOG(a)      ndarray_fp128_log(a)
#  define NDARRAY_FP128_LOG1P(a)    ndarray_fp128_log1p(a)
#  define NDARRAY_FP128_LOG2(a)     ndarray_fp128_log2(a)
#  define NDARRAY_FP128_LOG10(a)    ndarray_fp128_log10(a)
#  define NDARRAY_FP128_LOGB(a)     ndarray_fp128_logb(a)
#  define NDARRAY_FP128_ISNAN(a)    ndarray_fp128_isnan(a)
#  define NDARRAY_FP128_FROM_D(d)   ((ndarray_fp128_t)(d))
#  define NDARRAY_FP128_FROM_LD(ld) ((ndarray_fp128_t)(ld))
#  define NDARRAY_FP128_TO_D(x)     ((double)(x))
#  define NDARRAY_FP128_ZERO()      ((ndarray_fp128_t)0)
#  define NDARRAY_FP128_ONE()       ((ndarray_fp128_t)1)
#  define NDARRAY_FP128_NAN()       ((ndarray_fp128_t)(0.0/0.0))
#  define NDARRAY_FP128_ADD(a, b)   ((a) + (b))
#  define NDARRAY_FP128_SUB(a, b)   ((a) - (b))
#  define NDARRAY_FP128_MUL(a, b)   ((a) * (b))
#  define NDARRAY_FP128_DIV(a, b)   ((a) / (b))
#  define NDARRAY_FP128_NEG(a)      (-(a))
#  define NDARRAY_FP128_ABS(a)      ((a) < (ndarray_fp128_t)0 ? -(a) : (a))
#  define NDARRAY_FP128_EQ(a, b)    ((a) == (b))
#  define NDARRAY_FP128_LT(a, b)    ((a) <  (b))
#  define NDARRAY_FP128_ISZERO(a)   ((a) == (ndarray_fp128_t)0)
#  define NDARRAY_FP128_FROM_I64(i) ((ndarray_fp128_t)(i))
#  define NDARRAY_FP128_TO_I64(x)   ((long long)(x))
#else
#  define NDARRAY_FP128_FROM_D(d)   ndarray_dd_from_double(d)
#  define NDARRAY_FP128_FROM_LD(ld) ndarray_dd_from_double((double)(ld))
#  define NDARRAY_FP128_TO_D(x)     ndarray_dd_to_double(x)
#  define NDARRAY_FP128_ZERO()      ndarray_dd_from_double(0.0)
#  define NDARRAY_FP128_ONE()       ndarray_dd_from_double(1.0)
#  define NDARRAY_FP128_NAN()       ndarray_dd_from_double(0.0/0.0)
#  define NDARRAY_FP128_ADD(a, b)   ndarray_dd_add((a), (b))
#  define NDARRAY_FP128_SUB(a, b)   ndarray_dd_sub((a), (b))
#  define NDARRAY_FP128_MUL(a, b)   ndarray_dd_mul((a), (b))
#  define NDARRAY_FP128_DIV(a, b)   ndarray_dd_div((a), (b))
#  define NDARRAY_FP128_NEG(a)      ndarray_dd_neg(a)
#  define NDARRAY_FP128_ABS(a)      ndarray_dd_abs(a)
#  define NDARRAY_FP128_SQRT(a)     ndarray_dd_sqrt(a)
#  define NDARRAY_FP128_SIN(a)      ndarray_dd_from_double(sin(ndarray_dd_to_double(a)))
   /* Transcendental fp128 ops on the DD backend route through `double`
      so the macro contract stays platform-portable; accuracy is the same
      tier as the GPU `dd_*` reference path that already promotes
      `dd → double → libm → dd` for sin/cos/exp/log. Linux GCC x86-64 with
      libquadmath is the only configuration that yields full 113-bit
      transcendentals — every other platform tops out at fp64 here. */
#  define NDARRAY_FP128_EXP(a)      ndarray_dd_from_double(exp(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_EXP2(a)     ndarray_dd_from_double(exp2(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_EXPM1(a)    ndarray_dd_from_double(expm1(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_LOG(a)      ndarray_dd_from_double(log(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_LOG1P(a)    ndarray_dd_from_double(log1p(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_LOG2(a)     ndarray_dd_from_double(log2(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_LOG10(a)    ndarray_dd_from_double(log10(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_LOGB(a)     ndarray_dd_from_double(logb(ndarray_dd_to_double(a)))
#  define NDARRAY_FP128_EQ(a, b)    (ndarray_dd_cmp((a), (b)) == 0)
#  define NDARRAY_FP128_LT(a, b)    (ndarray_dd_cmp((a), (b)) <  0)
#  define NDARRAY_FP128_ISZERO(a)   ndarray_dd_iszero(a)
#  define NDARRAY_FP128_ISNAN(a)    ndarray_dd_isnan(a)
#  define NDARRAY_FP128_FROM_I64(i) ndarray_dd_from_int64(i)
#  define NDARRAY_FP128_TO_I64(x)   ndarray_dd_to_int64(x)
#endif

ndarray_fp128_t ndarray_double_to_fp128(double val);
ndarray_fp128_t ndarray_ldouble_to_fp128(long double val);
double          ndarray_fp128_to_double(ndarray_fp128_t val);
void            ndarray_fp128_to_string(ndarray_fp128_t val, char *buf, size_t bufsize);
ndarray_fp128_t ndarray_string_to_fp128(const char *str);

/* ── Generic element I/O ─────────────────────────────────────────────────── */

/* Format the element at [byte_offset] bytes into [data] as a C string.
   [type] is one of the NDARRAY_TYPE_* constants. [buf] must be at least 48
   bytes. */
void ndarray_element_to_string(const char *type,
                                const char *data,
                                size_t      byte_offset,
                                char       *buf,
                                size_t      bufsize);

/* Store [val] (as double) at element index [index] in [data].
   Performs the appropriate cast / quantisation for the target type.     */
void ndarray_set_from_double(const char *type, char *data, size_t index, double val);

/* Store a value parsed from string [str] at element index [index] in [data].
   Preferred over ndarray_set_from_double for float128 and uint64 to avoid
   precision loss.                                                           */
void ndarray_set_from_string(const char *type, char *data, size_t index, const char *str);

/* Read the element at element index [index] in [data] as a double. */
double ndarray_element_to_double(const char *type, const char *data, size_t index);

#endif /* PHPSCI_NDARRAY_TYPES_EXTRA_H */
