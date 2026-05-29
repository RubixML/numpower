#ifndef PHPSCI_NDARRAY_DOUBLE_MATH_H
#define PHPSCI_NDARRAY_DOUBLE_MATH_H

#ifndef _MSC_VER
#include "../../config.h"
#endif
#include "../ndarray.h"

/* `float_abs` / `float_sqrt` are kept for ad-hoc usage elsewhere. Every
   element-wise math method now dispatches through the typed unary / binary
   paths: trig / hyperbolic / angle / rounding via `NDArray_TypedUnaryOp`
   (`round` included, with precision support), `arctan2` via the typed
   binary dispatch (`NDArray_Arctan2_*` / `cuda_atan2_*`). */
float float_abs(float val);
float float_sqrt(float val);
#endif //PHPSCI_NDARRAY_DOUBLE_MATH_H
