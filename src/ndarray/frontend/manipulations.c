// HAVE_CUBLAS
#ifndef _MSC_VER
#include "../../../config.h"
#endif

#include <Zend/zend_interfaces.h>
#include <stdio.h>
#include <string.h>

// NDARRAY_TYPE_FLOAT32, NDARRAY_TYPE_FLOAT64, …
#include "../../types.h"

// NDArray, NDArray_TYPE
#include "../../ndarray.h"

// ndarray_set_from_double, ndarray_set_from_string
#include "../../ndarray_types.h"

#include "manipulations.h"

/**
 * @brief Encode @p value into one element of @p dtype.
 *
 * See `manipulations.h` for the full contract. Shared by
 * `NDArray_fillByZval()` (the in-place `fill()` path) and by every
 * factory method that accepts a typed PHP scalar as the fill value
 * (currently `NumPower::full`).
 */
int NDArray_EncodeZvalToDtype(zval *value, const char *dtype, char *out_buffer)
{
    /* Wider-than-double dtypes can't be expressed exactly in IS_LONG/IS_DOUBLE,
       so for IS_LONG inputs we route the value through strtoll/strtoull/
       strtoflt128 via a stringification step to preserve full bits. */
    int wants_string_long_path = (!strcmp(dtype, "int64")    ||
                                  !strcmp(dtype, "uint64")   ||
                                  !strcmp(dtype, "float128"));

    if (Z_TYPE_P(value) == IS_STRING) {
        ndarray_set_from_string(dtype, out_buffer, 0, Z_STRVAL_P(value));
        return 1;
    }
    if (Z_TYPE_P(value) == IS_LONG) {
        zend_long lv = Z_LVAL_P(value);
        if (wants_string_long_path) {
            /* PHP_INT_MAX (2^63 - 1) overflows the float64 mantissa; route
               through the string path so int64/uint64/fp128 keep all 64
               source bits. */
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%lld", (long long)lv);
            ndarray_set_from_string(dtype, out_buffer, 0, tmp);
        } else {
            ndarray_set_from_double(dtype, out_buffer, 0, (double)lv);
        }
        return 1;
    }
    if (Z_TYPE_P(value) == IS_DOUBLE) {
        ndarray_set_from_double(dtype, out_buffer, 0, Z_DVAL_P(value));
        return 1;
    }
    if (Z_TYPE_P(value) == IS_TRUE) {
        ndarray_set_from_double(dtype, out_buffer, 0, 1.0);
        return 1;
    }
    if (Z_TYPE_P(value) == IS_FALSE) {
        ndarray_set_from_double(dtype, out_buffer, 0, 0.0);
        return 1;
    }
    zend_throw_error(NULL,
        "Invalid value type. Supported types are: float, int, bool, string");
    return 0;
}

/**
 * @brief Fill every element of @p ndarray with @p value in the target's dtype.
 *
 * Supports the same PHP scalar types as offsetSet (string / int / double /
 * bool) and every NDArray dtype (float4..float128, int8..uint64). For dtypes
 * where the PHP scalar can't carry full precision (float128, int64, uint64)
 * IS_LONG values route through strtoll/strtoull/strtoflt128 — same precision
 * discipline as the offsetSet fast path.
 *
 * Implementation: encode the value once via `NDArray_EncodeZvalToDtype`,
 * then broadcast it across the target. On CPU the broadcast writes
 * directly into `NDArray_DATA`; on GPU the host stages an N-element buffer
 * and hands it to `NDArray_TypedH2D` (which converts __float128/DD host
 * bytes into the on-device DD layout for fp128). The staging buffer is
 * freed on every exit path.
 *
 * @param[in,out] ndarray Target NDArray (CPU or GPU resident).
 * @param[in]     value   PHP scalar broadcast across @p ndarray.
 */
void NDArray_fillByZval(NDArray *ndarray, zval *value)
{
    if (ndarray == NULL) {
        zend_throw_error(NULL, "Invalid NDArray");
        return;
    }

    const char *dtype  = NDArray_TYPE(ndarray);
    long        n      = NDArray_NUMELEMENTS(ndarray);
    int         elsize = NDArray_ELSIZE(ndarray);

    if (n <= 0) return;

    /* Encode the scalar into a 16-byte scratch buffer once. 16 bytes = the
       widest dtype (fp128); narrower dtypes use the leading elsize bytes.
       The scratch is zero-initialised so any unused tail bytes (e.g. when
       elsize==1 for fp4) stay deterministic. */
    char encoded[16];
    memset(encoded, 0, sizeof(encoded));
    if (!NDArray_EncodeZvalToDtype(value, dtype, encoded)) {
        return;
    }

    /* Pick the broadcast target: directly into NDArray_DATA on CPU, into a
       host staging buffer on GPU. */
    char *target_data;
#ifdef HAVE_CUBLAS
    char *gpu_tmp = NULL;
    if (NDArray_DEVICE(ndarray) == NDARRAY_DEVICE_GPU) {
        gpu_tmp = emalloc((size_t)n * (size_t)elsize);
        target_data = gpu_tmp;
    } else {
        target_data = (char *)NDArray_DATA(ndarray);
    }
#else
    target_data = (char *)NDArray_DATA(ndarray);
#endif

    /* Broadcast `encoded[0..elsize-1]` across `target_data[0..n*elsize-1]`.
       For 1-byte dtypes (fp4/fp8/int8/uint8) memset is a single SIMD-friendly
       call; wider dtypes go through a memcpy loop. */
    if (elsize == 1) {
        memset(target_data, encoded[0], (size_t)n);
    } else {
        for (long i = 0; i < n; i++) {
            memcpy(target_data + (size_t)i * (size_t)elsize, encoded, (size_t)elsize);
        }
    }

#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(ndarray) == NDARRAY_DEVICE_GPU) {
        /* fp128 staging is __float128/dd on host, but lands on GPU as the
           double-double (hi, lo) pair — TypedH2D handles the conversion. */
        NDArray_TypedH2D((char *)NDArray_DATA(ndarray), gpu_tmp, n, dtype);
        efree(gpu_tmp);
    }
#endif
}
