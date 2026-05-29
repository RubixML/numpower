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

