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

/* Fill every element of [ndarray] with [value] in the target's dtype.
   Supports the same PHP scalar types as offsetSet (string / int / double /
   bool) and every NDArray dtype (float4..float128, int8..uint64). For dtypes
   where the PHP scalar can't carry full precision (float128, int64, uint64),
   IS_LONG values route through strtoll/strtoull/strtoflt128 instead of double
   — same precision discipline as the offsetSet fast path.

   Implementation note: encode the value once into a 16-byte scratch buffer
   (max elsize across dtypes is 16 for fp128) then broadcast it across the
   target. Avoids the per-element strcmp dispatch chain inside
   ndarray_set_from_*, which would be O(N*K) for an N-element fill.

   On GPU the broadcast target is a host-side staging buffer of [n * elsize]
   bytes which is then handed to NDArray_TypedH2D — that helper knows how to
   convert __float128 / double-double host bytes to the on-device dd layout.
   The staging buffer is freed on every exit path. */
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

    /* Wider-than-double dtypes can't be expressed exactly in IS_LONG/IS_DOUBLE,
       so for IS_LONG inputs we route the value through strtoll/strtoull/
       strtoflt128 via a stringification step to preserve full bits. */
    int wants_string_long_path = (!strcmp(dtype, "int64")    ||
                                  !strcmp(dtype, "uint64")   ||
                                  !strcmp(dtype, "float128"));

    /* Encode the scalar into a 16-byte scratch buffer once. 16 bytes = the
       widest dtype (fp128); narrower dtypes use the leading elsize bytes.
       The scratch is zero-initialised so any unused tail bytes (e.g. when
       elsize==1 for fp4) stay deterministic. */
    char encoded[16];
    memset(encoded, 0, sizeof(encoded));

    if (Z_TYPE_P(value) == IS_STRING) {
        ndarray_set_from_string(dtype, encoded, 0, Z_STRVAL_P(value));
    } else if (Z_TYPE_P(value) == IS_LONG) {
        zend_long lv = Z_LVAL_P(value);
        if (wants_string_long_path) {
            /* PHP_INT_MAX (2^63 - 1) overflows the float64 mantissa; route
               through the string path so int64/uint64/fp128 keep all 64
               source bits. */
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%lld", (long long)lv);
            ndarray_set_from_string(dtype, encoded, 0, tmp);
        } else {
            ndarray_set_from_double(dtype, encoded, 0, (double)lv);
        }
    } else if (Z_TYPE_P(value) == IS_DOUBLE) {
        ndarray_set_from_double(dtype, encoded, 0, Z_DVAL_P(value));
    } else if (Z_TYPE_P(value) == IS_TRUE) {
        ndarray_set_from_double(dtype, encoded, 0, 1.0);
    } else if (Z_TYPE_P(value) == IS_FALSE) {
        ndarray_set_from_double(dtype, encoded, 0, 0.0);
    } else {
        zend_throw_error(NULL, "Invalid value type. Supported types are: float, int, bool, string");
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
