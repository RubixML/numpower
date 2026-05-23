#ifndef PHPSCI_NDARRAY_MANIPULATIONS_H
#define PHPSCI_NDARRAY_MANIPULATIONS_H

#include "../../ndarray.h"

/**
 * @brief Encode a PHP scalar @p value into one element of @p dtype.
 *
 * Writes `get_type_size(dtype)` bytes into @p out_buffer (caller must
 * supply at least 16 bytes — the widest dtype, fp128, is 16 bytes).
 * Supports IS_LONG, IS_DOUBLE, IS_STRING, IS_TRUE, IS_FALSE inputs. For
 * the "string-IO" dtypes (`float128`, `int64`, `uint64`) a numeric long
 * input is stringified first so that values outside PHP's native long
 * range round-trip byte-correctly through `ndarray_set_from_string`.
 * For dtypes that don't need string IO the IS_LONG input goes through
 * `ndarray_set_from_double`.
 *
 * On rejection the function throws a catchable `Error` and returns 0.
 *
 * @param[in]  value      PHP scalar; only IS_LONG/IS_DOUBLE/IS_STRING/
 *                        IS_TRUE/IS_FALSE produce a successful encode.
 * @param[in]  dtype      Canonical NDArray dtype string.
 * @param[out] out_buffer ≥ 16 bytes; receives the encoded value.
 * @return 1 on success, 0 on rejection (Error in flight).
 */
int NDArray_EncodeZvalToDtype(zval *value, const char *dtype, char *out_buffer);

void NDArray_fillByZval(NDArray *ndarray, zval *value);

#endif //PHPSCI_NDARRAY_MANIPULATIONS_H