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

/* ── float128 : __float128 on GCC/x86-64, long double elsewhere ─────────── */
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#  define NDARRAY_HAVE_FLOAT128 1
   typedef __float128 ndarray_fp128_t;
#  define NDARRAY_FP128_SIZE 16
#else
#  define NDARRAY_HAVE_FLOAT128 0
   typedef long double ndarray_fp128_t;
#  define NDARRAY_FP128_SIZE sizeof(long double)
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
