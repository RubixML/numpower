#include "double_math.h"
#include <math.h>
#ifndef _MSC_VER
#include "../../config.h"
#endif

/**
 * @brief Compute `|val|` at float precision.
 *
 * Kept for ad-hoc callers that thread a function pointer through the
 * legacy element-wise paths; the typed unary dispatcher routes
 * `NumPower::abs` directly through `cuda_abs_*` / inline libm now.
 *
 * @param[in] val Input.
 * @return `fabsf(val)`.
 */
float
float_abs(float val) {
    return fabsf(val);
}

/**
 * @brief Compute `sqrt(val)` at float precision.
 *
 * Kept for ad-hoc callers that thread a function pointer through the
 * legacy element-wise paths; the typed unary dispatcher routes
 * `NumPower::sqrt` directly through `cuda_sqrt_*` / inline libm now.
 *
 * @param[in] val Input.
 * @return `sqrtf(val)`.
 */
float
float_sqrt(float val) {
    return sqrtf(val);
}

/**
 * @brief Round @p number to @p decimals decimal places.
 *
 * Kept on the legacy `NDArray_Map1F` path because `NumPower::round`
 * accepts an extra precision argument that the typed unary
 * dispatcher does not yet support. Like every legacy `NDArray_Map*`
 * caller, this silently truncates non-fp32 inputs to fp32 — a known
 * carry-over from the trig family bug fix that is out of scope here.
 *
 * @param[in] number   Input value.
 * @param[in] decimals Number of decimal places to keep.
 * @return `roundf(number * 10^decimals) / 10^decimals`.
 */
float float_round(float number, float decimals) {
    float factor = powf(10, decimals);
    return roundf(number * factor) / factor;
}
