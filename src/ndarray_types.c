#include "ndarray_types.h"
#ifndef _MSC_VER
#include "../config.h"
#endif
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#if HAVE_QUADMATH && NDARRAY_HAVE_FLOAT128
#  include <quadmath.h>
#endif

/* ══════════════════════════════════════════════════════════════════════════
   float4 – E2M1 format (1 sign | 2 exponent | 1 mantissa), bias = 1
   16 representable values, stored one per byte (lower nibble).
   ══════════════════════════════════════════════════════════════════════════ */

static const double FP4_LOOKUP[16] = {
    /* 0b0000 */  0.0,
    /* 0b0001 */  0.5,
    /* 0b0010 */  1.0,
    /* 0b0011 */  1.5,
    /* 0b0100 */  2.0,
    /* 0b0101 */  3.0,
    /* 0b0110 */  4.0,
    /* 0b0111 */  6.0,
    /* 0b1000 */ -0.0,
    /* 0b1001 */ -0.5,
    /* 0b1010 */ -1.0,
    /* 0b1011 */ -1.5,
    /* 0b1100 */ -2.0,
    /* 0b1101 */ -3.0,
    /* 0b1110 */ -4.0,
    /* 0b1111 */ -6.0,
};

double ndarray_fp4_to_double(uint8_t nibble) {
    return FP4_LOOKUP[nibble & 0x0F];
}

uint8_t ndarray_double_to_fp4(double val) {
    uint8_t best = 0;
    double  best_err = fabs(val - FP4_LOOKUP[0]);
    for (uint8_t i = 1; i < 16; i++) {
        double err = fabs(val - FP4_LOOKUP[i]);
        if (err < best_err) {
            best_err = err;
            best     = i;
        }
    }
    return best;
}

/* ══════════════════════════════════════════════════════════════════════════
   float8 – E4M3 format (1 sign | 4 exponent | 3 mantissa), bias = 7
   exp=15 → NaN (any mantissa).  Max normal value ≈ 240.
   ══════════════════════════════════════════════════════════════════════════ */

double ndarray_fp8_to_double(uint8_t fp8) {
    int sign = (fp8 >> 7) & 1;
    int exp  = (fp8 >> 3) & 0xF;
    int man  = fp8 & 0x7;

    if (exp == 0xF) return NAN;

    double val;
    if (exp == 0) {
        val = ((double)man / 8.0) * ldexp(1.0, -6);
    } else {
        val = (1.0 + (double)man / 8.0) * ldexp(1.0, exp - 7);
    }
    return sign ? -val : val;
}

uint8_t ndarray_double_to_fp8(double val) {
    if (isnan(val)) return 0xFF;

    int    sign = (val < 0.0 || (val == 0.0 && (1.0 / val) < 0)) ? 1 : 0;
    double ax   = fabs(val);

    if (ax == 0.0) return (uint8_t)(sign << 7);

    /* Clamp to max representable: (1 + 7/8) * 2^7 = 240 */
    if (ax > 240.0) ax = 240.0;

    int    exp;
    double frac = frexp(ax, &exp); /* ax = frac * 2^exp, frac ∈ [0.5, 1.0) */
    /* Rewrite: ax = (2*frac) * 2^(exp-1) → biased_exp = (exp-1) + 7 = exp+6 */
    int biased_exp = exp + 6;

    if (biased_exp <= 0) {
        /* Subnormal: val = (man/8) * 2^(-6) → man = ax * 8 * 2^6 = ax * 512 */
        int man = (int)(ax * 512.0 + 0.5);
        if (man > 7) man = 7;
        return (uint8_t)((sign << 7) | man);
    }

    if (biased_exp >= 15) {
        /* Clamp to max: exp=14, man=7 → (1+7/8)*2^7 = 240 */
        return (uint8_t)((sign << 7) | (14 << 3) | 7);
    }

    /* Normal: 1.mmm * 2^(biased_exp-7); 2*frac ∈ [1.0, 2.0) → man = (2*frac-1)*8 */
    int man = (int)((2.0 * frac - 1.0) * 8.0 + 0.5);
    if (man > 7) {
        man = 0;
        biased_exp++;
    }
    if (biased_exp >= 15) {
        return (uint8_t)((sign << 7) | (14 << 3) | 7);
    }
    return (uint8_t)((sign << 7) | ((biased_exp & 0xF) << 3) | (man & 0x7));
}

/* ══════════════════════════════════════════════════════════════════════════
   float16 – IEEE-754 half precision  (1 sign | 5 exponent | 10 mantissa)
   ══════════════════════════════════════════════════════════════════════════ */

double ndarray_fp16_to_double(uint16_t fp16) {
    uint32_t sign = (uint32_t)(fp16 >> 15) & 1u;
    uint32_t exp  = (uint32_t)(fp16 >> 10) & 0x1Fu;
    uint32_t man  = (uint32_t)(fp16)        & 0x3FFu;

    uint32_t f32;
    if (exp == 31u) {
        /* Inf / NaN */
        f32 = (sign << 31) | 0x7F800000u | (man << 13);
    } else if (exp == 0u) {
        if (man == 0u) {
            f32 = sign << 31;
        } else {
            /* Subnormal fp16 → normalised fp32 */
            exp = 1u;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            f32 = (sign << 31) | ((exp + 127u - 15u) << 23) | (man << 13);
        }
    } else {
        f32 = (sign << 31) | ((exp + 127u - 15u) << 23) | (man << 13);
    }

    float fval;
    memcpy(&fval, &f32, 4);
    return (double)fval;
}

uint16_t ndarray_double_to_fp16(double val) {
    float f = (float)val;
    uint32_t u;
    memcpy(&u, &f, 4);

    uint32_t sign = u >> 31;
    uint32_t u_exp = (u >> 23) & 0xFFu;
    uint32_t man  = (u >> 13) & 0x3FFu; /* top 10 bits of 23-bit mantissa */

    if (u_exp == 255u) {
        /* Inf or NaN */
        return (uint16_t)((sign << 15) | 0x7C00u | (man ? 0x0200u : 0u));
    }

    int exp32 = (int)u_exp - 127 + 15; /* rebias for fp16 */

    if (exp32 <= 0) {
        if (exp32 < -10) return (uint16_t)(sign << 15); /* underflow → ±0 */
        /* Subnormal fp16 */
        uint32_t m = ((u >> 23) & 0xFFu) ? ((u & 0x7FFFFFu) | 0x800000u) : 0u;
        m >>= (1 - exp32 + 13);
        return (uint16_t)((sign << 15) | (m & 0x3FFu));
    }

    if (exp32 >= 31) return (uint16_t)((sign << 15) | 0x7C00u); /* Inf */

    return (uint16_t)((sign << 15) | ((uint32_t)exp32 << 10) | man);
}

/* ══════════════════════════════════════════════════════════════════════════
   float128 — __float128+libquadmath on Linux GCC x86, otherwise double-double
   (see src/dd_math.h). String I/O always preserves the full precision the
   chosen storage can represent.
   ══════════════════════════════════════════════════════════════════════════ */

ndarray_fp128_t ndarray_double_to_fp128(double val) {
    return NDARRAY_FP128_FROM_D(val);
}

ndarray_fp128_t ndarray_ldouble_to_fp128(long double val) {
    return NDARRAY_FP128_FROM_LD(val);
}

double ndarray_fp128_to_double(ndarray_fp128_t val) {
    return NDARRAY_FP128_TO_D(val);
}

void ndarray_fp128_to_string(ndarray_fp128_t val, char *buf, size_t bufsize) {
#if HAVE_QUADMATH && NDARRAY_HAVE_FLOAT128
    /* libquadmath supplies %Qg which renders all 113 mantissa bits (~34
       decimal digits). Use it when available so fp128 round-trips show the
       full precision the CPU compute actually retains. */
    quadmath_snprintf(buf, bufsize, "%.34Qg", val);
#elif NDARRAY_HAVE_FLOAT128
    /* libquadmath not present: long double has ~18-19 significant digits. */
    long double ld = (long double)val;
    snprintf(buf, bufsize, "%.18Lg", ld);
#else
    /* DD path: dedicated formatter that walks the (hi, lo) pair to recover
       the full ~32 decimal digits worth of mantissa. */
    ndarray_dd_to_string(val, buf, bufsize);
#endif
}

ndarray_fp128_t ndarray_string_to_fp128(const char *str) {
#if HAVE_QUADMATH && NDARRAY_HAVE_FLOAT128
    char *endptr;
    return strtoflt128(str, &endptr);
#elif NDARRAY_HAVE_FLOAT128
    char *endptr;
    long double ld = strtold(str, &endptr);
    return (ndarray_fp128_t)ld;
#else
    /* DD path: digit-accumulating parser so the lo word captures the bits
       beyond fp64 precision. strtold here would silently truncate to ~17
       digits on Apple Silicon / MSVC where long double == double. */
    return ndarray_dd_from_string(str);
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
   Generic element I/O
   ══════════════════════════════════════════════════════════════════════════ */

void ndarray_element_to_string(const char *type,
                                const char *data,
                                size_t      byte_offset,
                                char       *buf,
                                size_t      bufsize)
{
    if (strcmp(type, "float4") == 0) {
        uint8_t nibble = (uint8_t)(((const uint8_t *)data)[byte_offset] & 0x0Fu);
        snprintf(buf, bufsize, "%g", ndarray_fp4_to_double(nibble));

    } else if (strcmp(type, "float8") == 0) {
        uint8_t fp8 = ((const uint8_t *)data)[byte_offset];
        double  v   = ndarray_fp8_to_double(fp8);
        if (isnan(v)) {
            snprintf(buf, bufsize, "nan");
        } else {
            snprintf(buf, bufsize, "%g", v);
        }

    } else if (strcmp(type, "float16") == 0) {
        uint16_t fp16;
        memcpy(&fp16, data + byte_offset, 2);
        snprintf(buf, bufsize, "%.6g", ndarray_fp16_to_double(fp16));

    } else if (strcmp(type, "float32") == 0) {
        float v;
        memcpy(&v, data + byte_offset, 4);
        snprintf(buf, bufsize, "%.8g", (double)v);

    } else if (strcmp(type, "float64") == 0) {
        double v;
        memcpy(&v, data + byte_offset, 8);
        snprintf(buf, bufsize, "%.16g", v);

    } else if (strcmp(type, "float128") == 0) {
        ndarray_fp128_t v;
        memcpy(&v, data + byte_offset, NDARRAY_FP128_SIZE);
        ndarray_fp128_to_string(v, buf, bufsize);

    } else if (strcmp(type, "int8") == 0) {
        int8_t v = (int8_t)((const uint8_t *)data)[byte_offset];
        snprintf(buf, bufsize, "%d", (int)v);

    } else if (strcmp(type, "uint8") == 0) {
        uint8_t v = ((const uint8_t *)data)[byte_offset];
        snprintf(buf, bufsize, "%u", (unsigned)v);

    } else if (strcmp(type, "int16") == 0) {
        int16_t v;
        memcpy(&v, data + byte_offset, 2);
        snprintf(buf, bufsize, "%d", (int)v);

    } else if (strcmp(type, "uint16") == 0) {
        uint16_t v;
        memcpy(&v, data + byte_offset, 2);
        snprintf(buf, bufsize, "%u", (unsigned)v);

    } else if (strcmp(type, "int32") == 0) {
        int32_t v;
        memcpy(&v, data + byte_offset, 4);
        snprintf(buf, bufsize, "%" PRId32, v);

    } else if (strcmp(type, "uint32") == 0) {
        uint32_t v;
        memcpy(&v, data + byte_offset, 4);
        snprintf(buf, bufsize, "%" PRIu32, v);

    } else if (strcmp(type, "int64") == 0) {
        int64_t v;
        memcpy(&v, data + byte_offset, 8);
        snprintf(buf, bufsize, "%" PRId64, v);

    } else if (strcmp(type, "uint64") == 0) {
        uint64_t v;
        memcpy(&v, data + byte_offset, 8);
        snprintf(buf, bufsize, "%" PRIu64, v);

    } else {
        snprintf(buf, bufsize, "?");
    }
}

void ndarray_set_from_double(const char *type, char *data, size_t index, double val)
{
    if (strcmp(type, "float4") == 0) {
        ((uint8_t *)data)[index] = ndarray_double_to_fp4(val);

    } else if (strcmp(type, "float8") == 0) {
        ((uint8_t *)data)[index] = ndarray_double_to_fp8(val);

    } else if (strcmp(type, "float16") == 0) {
        uint16_t v = ndarray_double_to_fp16(val);
        memcpy((uint16_t *)data + index, &v, 2);

    } else if (strcmp(type, "float32") == 0) {
        float v = (float)val;
        memcpy((float *)data + index, &v, 4);

    } else if (strcmp(type, "float64") == 0) {
        memcpy((double *)data + index, &val, 8);

    } else if (strcmp(type, "float128") == 0) {
        ndarray_fp128_t v = ndarray_double_to_fp128(val);
        memcpy((ndarray_fp128_t *)data + index, &v, NDARRAY_FP128_SIZE);

    } else if (strcmp(type, "int8") == 0) {
        ((int8_t *)data)[index] = (int8_t)(int64_t)val;

    } else if (strcmp(type, "uint8") == 0) {
        ((uint8_t *)data)[index] = (uint8_t)(uint64_t)val;

    } else if (strcmp(type, "int16") == 0) {
        int16_t v = (int16_t)(int64_t)val;
        memcpy((int16_t *)data + index, &v, 2);

    } else if (strcmp(type, "uint16") == 0) {
        uint16_t v = (uint16_t)(uint64_t)val;
        memcpy((uint16_t *)data + index, &v, 2);

    } else if (strcmp(type, "int32") == 0) {
        int32_t v = (int32_t)(int64_t)val;
        memcpy((int32_t *)data + index, &v, 4);

    } else if (strcmp(type, "uint32") == 0) {
        uint32_t v = (uint32_t)(uint64_t)val;
        memcpy((uint32_t *)data + index, &v, 4);

    } else if (strcmp(type, "int64") == 0) {
        int64_t v = (int64_t)val;
        memcpy((int64_t *)data + index, &v, 8);

    } else if (strcmp(type, "uint64") == 0) {
        uint64_t v = (uint64_t)val;
        memcpy((uint64_t *)data + index, &v, 8);
    }
}

double ndarray_element_to_double(const char *type, const char *data, size_t index)
{
    if (strcmp(type, "float4") == 0) {
        uint8_t nibble = (uint8_t)(((const uint8_t *)data)[index] & 0x0Fu);
        return ndarray_fp4_to_double(nibble);
    } else if (strcmp(type, "float8") == 0) {
        uint8_t fp8 = ((const uint8_t *)data)[index];
        return ndarray_fp8_to_double(fp8);
    } else if (strcmp(type, "float16") == 0) {
        uint16_t fp16;
        memcpy(&fp16, (const uint16_t *)data + index, 2);
        return ndarray_fp16_to_double(fp16);
    } else if (strcmp(type, "float32") == 0) {
        float v;
        memcpy(&v, (const float *)data + index, 4);
        return (double)v;
    } else if (strcmp(type, "float64") == 0) {
        double v;
        memcpy(&v, (const double *)data + index, 8);
        return v;
    } else if (strcmp(type, "float128") == 0) {
        ndarray_fp128_t v;
        memcpy(&v, (const ndarray_fp128_t *)data + index, NDARRAY_FP128_SIZE);
        return ndarray_fp128_to_double(v);
    } else if (strcmp(type, "int8") == 0) {
        return (double)(int8_t)((const uint8_t *)data)[index];
    } else if (strcmp(type, "uint8") == 0) {
        return (double)((const uint8_t *)data)[index];
    } else if (strcmp(type, "int16") == 0) {
        int16_t v;
        memcpy(&v, (const int16_t *)data + index, 2);
        return (double)v;
    } else if (strcmp(type, "uint16") == 0) {
        uint16_t v;
        memcpy(&v, (const uint16_t *)data + index, 2);
        return (double)v;
    } else if (strcmp(type, "int32") == 0) {
        int32_t v;
        memcpy(&v, (const int32_t *)data + index, 4);
        return (double)v;
    } else if (strcmp(type, "uint32") == 0) {
        uint32_t v;
        memcpy(&v, (const uint32_t *)data + index, 4);
        return (double)v;
    } else if (strcmp(type, "int64") == 0) {
        int64_t v;
        memcpy(&v, (const int64_t *)data + index, 8);
        return (double)v;
    } else if (strcmp(type, "uint64") == 0) {
        uint64_t v;
        memcpy(&v, (const uint64_t *)data + index, 8);
        return (double)v;
    }
    return 0.0;
}

void ndarray_set_from_string(const char *type, char *data, size_t index, const char *str)
{
    if (strcmp(type, "float128") == 0) {
        ndarray_fp128_t v = ndarray_string_to_fp128(str);
        memcpy((ndarray_fp128_t *)data + index, &v, NDARRAY_FP128_SIZE);

    } else if (strcmp(type, "uint64") == 0) {
        char    *endptr;
        uint64_t v = (uint64_t)strtoull(str, &endptr, 10);
        memcpy((uint64_t *)data + index, &v, 8);

    } else if (strcmp(type, "int64") == 0) {
        char   *endptr;
        int64_t v = (int64_t)strtoll(str, &endptr, 10);
        memcpy((int64_t *)data + index, &v, 8);

    } else {
        char   *endptr;
        double  v = strtod(str, &endptr);
        ndarray_set_from_double(type, data, index, v);
    }
}
