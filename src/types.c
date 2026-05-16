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
       for full fidelity). */
    if (!strcmp(type, "float4"))   return 1;
    if (!strcmp(type, "float8"))   return 1;
    if (!strcmp(type, "float16"))  return 1;
    if (!strcmp(type, "float128")) return 1;
    if (!strcmp(type, "uint64"))   return 1;
    return 0;
}

int type_is_valid(const char *type) {
    return get_type_size(type) > 0;
}
