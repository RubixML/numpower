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

const char *promote_dtype(const char *a, const char *b)
{
    /* Float promotion hierarchy matching PyTorch */
    static const char *const ranks[] = {
        "float4", "float8", "float16", "float32", "float64", "float128", NULL
    };
    int ra = -1, rb = -1;
    for (int i = 0; ranks[i] != NULL; i++) {
        if (!strcmp(a, ranks[i])) ra = i;
        if (!strcmp(b, ranks[i])) rb = i;
    }
    /* If types not found in float hierarchy, return 'a' as fallback */
    if (ra < 0 && rb < 0) return a;
    if (ra < 0) return b;
    if (rb < 0) return a;
    return (ra >= rb) ? a : b;
}

const char *compute_dtype_for_arithmetic(const char *result_dtype)
{
    /* float16 and below need to be computed in float32 since there's no native CPU fp16 arithmetic */
    if (!strcmp(result_dtype, "float4")  ||
        !strcmp(result_dtype, "float8")  ||
        !strcmp(result_dtype, "float16")) {
        return "float32";
    }
    return result_dtype;
}
