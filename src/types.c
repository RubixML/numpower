#include "types.h"
#include "ndarray_types.h"
#include <string.h>

int get_type_size(const char *type) {
    if (!strcmp(type, "float4"))   return 1;   /* 4 bits in lower nibble of uint8_t */
    if (!strcmp(type, "float8"))   return 1;
    if (!strcmp(type, "float16"))  return 2;
    if (!strcmp(type, "float32"))  return 4;
    if (!strcmp(type, "float64"))  return 8;
    if (!strcmp(type, "float128")) return (int)NDARRAY_FP128_SIZE;
    if (!strcmp(type, "int8"))     return 1;
    if (!strcmp(type, "uint8"))    return 1;
    if (!strcmp(type, "int16"))    return 2;
    if (!strcmp(type, "uint16"))   return 2;
    if (!strcmp(type, "int32"))    return 4;
    if (!strcmp(type, "uint32"))   return 4;
    if (!strcmp(type, "int64"))    return 8;
    if (!strcmp(type, "uint64"))   return 8;
    return 0;
}

int is_type(const char *type_a, const char *type_b) {
    return strcmp(type_a, type_b) == 0 ? 1 : 0;
}

int type_needs_string_io(const char *type) {
    /* Types not natively representable in PHP (int64/uint64 may need strings
       for values outside PHP_INT range; float4/8/16/128 always need strings
       for full fidelity). int64 is included so that string inputs like
       "9223372036854775807" go through strtoll instead of being routed via
       double, which silently rounds away the bottom bits past 2^53. */
    if (!strcmp(type, "float4"))   return 1;
    if (!strcmp(type, "float8"))   return 1;
    if (!strcmp(type, "float16"))  return 1;
    if (!strcmp(type, "float128")) return 1;
    if (!strcmp(type, "int64"))    return 1;
    if (!strcmp(type, "uint64"))   return 1;
    return 0;
}

int type_is_valid(const char *type) {
    return get_type_size(type) > 0;
}

const char *type_canonicalize(const char *type) {
    if (type == NULL) return NULL;
    if (!strcmp(type, "float4"))   return NDARRAY_TYPE_FLOAT4;
    if (!strcmp(type, "float8"))   return NDARRAY_TYPE_FLOAT8;
    if (!strcmp(type, "float16"))  return NDARRAY_TYPE_FLOAT16;
    if (!strcmp(type, "float32"))  return NDARRAY_TYPE_FLOAT32;
    if (!strcmp(type, "float64"))  return NDARRAY_TYPE_FLOAT64;
    if (!strcmp(type, "float128")) return NDARRAY_TYPE_FLOAT128;
    if (!strcmp(type, "int8"))     return NDARRAY_TYPE_INT8;
    if (!strcmp(type, "uint8"))    return NDARRAY_TYPE_UINT8;
    if (!strcmp(type, "int16"))    return NDARRAY_TYPE_INT16;
    if (!strcmp(type, "uint16"))   return NDARRAY_TYPE_UINT16;
    if (!strcmp(type, "int32"))    return NDARRAY_TYPE_INT32;
    if (!strcmp(type, "uint32"))   return NDARRAY_TYPE_UINT32;
    if (!strcmp(type, "int64"))    return NDARRAY_TYPE_INT64;
    if (!strcmp(type, "uint64"))   return NDARRAY_TYPE_UINT64;
    return NULL;
}

/* Category constants for promote_dtype */
#define DTYPE_CAT_UNKNOWN  0
#define DTYPE_CAT_UINT     1
#define DTYPE_CAT_INT      2
#define DTYPE_CAT_FLOAT    3

static int dtype_category(const char *t) {
    if (!strcmp(t, "uint8")  || !strcmp(t, "uint16") ||
        !strcmp(t, "uint32") || !strcmp(t, "uint64")) return DTYPE_CAT_UINT;
    if (!strcmp(t, "int8")   || !strcmp(t, "int16")  ||
        !strcmp(t, "int32")  || !strcmp(t, "int64"))  return DTYPE_CAT_INT;
    if (!strcmp(t, "float4")  || !strcmp(t, "float8")  || !strcmp(t, "float16") ||
        !strcmp(t, "float32") || !strcmp(t, "float64") || !strcmp(t, "float128"))
        return DTYPE_CAT_FLOAT;
    return DTYPE_CAT_UNKNOWN;
}

/* Width in bits; used inside the same numeric category to pick the wider type. */
static int dtype_width_bits(const char *t) {
    if (!strcmp(t, "float4"))   return 4;
    if (!strcmp(t, "float8"))   return 8;
    if (!strcmp(t, "float16"))  return 16;
    if (!strcmp(t, "float32"))  return 32;
    if (!strcmp(t, "float64"))  return 64;
    if (!strcmp(t, "float128")) return 128;
    if (!strcmp(t, "int8"))     return 8;
    if (!strcmp(t, "uint8"))    return 8;
    if (!strcmp(t, "int16"))    return 16;
    if (!strcmp(t, "uint16"))   return 16;
    if (!strcmp(t, "int32"))    return 32;
    if (!strcmp(t, "uint32"))   return 32;
    if (!strcmp(t, "int64"))    return 64;
    if (!strcmp(t, "uint64"))   return 64;
    return 0;
}

const char *promote_dtype(const char *a, const char *b)
{
    /* Type promotion matching PyTorch:
       - float wins over int/uint
       - within float / int / uint, the wider type wins
       - int8 + uint8 → int16 (matches PyTorch)
       - intN + uintN → intM where M = N*2 (e.g. int32 + uint32 → int64).
         For int64 + uint64 there is no wider int that fits both; PyTorch
         picks float32 in that pathological case.
       - Equal types → return that type.                                       */
    if (!strcmp(a, b)) return a;

    int ca = dtype_category(a), cb = dtype_category(b);

    /* Mixed unknown → fall back to a. */
    if (ca == DTYPE_CAT_UNKNOWN && cb == DTYPE_CAT_UNKNOWN) return a;
    if (ca == DTYPE_CAT_UNKNOWN) return b;
    if (cb == DTYPE_CAT_UNKNOWN) return a;

    /* Float wins over int/uint regardless of width. */
    if (ca == DTYPE_CAT_FLOAT && cb != DTYPE_CAT_FLOAT) return a;
    if (cb == DTYPE_CAT_FLOAT && ca != DTYPE_CAT_FLOAT) return b;

    /* Same category → pick wider. */
    if (ca == cb) {
        int wa = dtype_width_bits(a), wb = dtype_width_bits(b);
        return wa >= wb ? a : b;
    }

    /* int + uint of same width: promote to next signed int wide enough to hold both. */
    int wa = dtype_width_bits(a), wb = dtype_width_bits(b);
    int wmax = wa > wb ? wa : wb;
    if (wa == wb) {
        switch (wmax) {
            case 8:  return "int16";
            case 16: return "int32";
            case 32: return "int64";
            case 64: return "float32"; /* nothing wider in int family; PyTorch picks float */
        }
    }
    /* int wider than uint, or vice versa: signed needs to fit, prefer the int type if wider. */
    if (ca == DTYPE_CAT_INT  && wa >  wb) return a;
    if (cb == DTYPE_CAT_INT  && wb >  wa) return b;
    /* uint wider than int → need signed type strictly wider than the uint. */
    int wuint = (ca == DTYPE_CAT_UINT) ? wa : wb;
    switch (wuint) {
        case 8:  return "int16";
        case 16: return "int32";
        case 32: return "int64";
        case 64: return "float32";
    }
    return a;
}

const char *compute_dtype_for_arithmetic(const char *result_dtype)
{
    /* Map any dtype to a compute type. The available compute kernels are:
       - float32 / float64 / float128 (the legacy float kernels), and
       - every native integer dtype (int8..int64, uint8..uint64) routed
         through `NDArray_TypedBinOp_CPU_Int` on CPU and the per-dtype
         `cuda_<op>_<tag>` kernels on GPU.

       Integer dtypes now keep their native compute type so PyTorch's
       modular wrap-around semantics survive end-to-end. The prior
       implementation promoted narrow ints to float32 and 32/64-bit ints
       to float64; once an intermediate exceeded the float mantissa
       (24 bits for narrow ints — still safe for int8..int16 — or 53 bits
       for the wider ints) the result silently rounded, and the cast back
       to the narrow target diverged between CPU (double round-trip wraps)
       and GPU (`cuda_cast_f64_to_i32` saturates). Native int compute
       eliminates both issues.

       Caller (`ndarray_promote_and_op`) casts the result back to
       result_dtype after the op runs whenever comp_type and result_type
       differ (e.g. for division which promotes to float). */
    if (!strcmp(result_dtype, "float128")) return "float128";
    if (!strcmp(result_dtype, "float64"))  return "float64";
    if (!strcmp(result_dtype, "int8")    || !strcmp(result_dtype, "uint8")  ||
        !strcmp(result_dtype, "int16")   || !strcmp(result_dtype, "uint16") ||
        !strcmp(result_dtype, "int32")   || !strcmp(result_dtype, "uint32") ||
        !strcmp(result_dtype, "int64")   || !strcmp(result_dtype, "uint64")) {
        return result_dtype;
    }
    /* float4, float8, float16, float32 → float32 (narrow floats route
       through the float32 kernel and the result casts back to the
       caller's dtype). */
    return "float32";
}
