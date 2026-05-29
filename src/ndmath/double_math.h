#ifndef PHPSCI_NDARRAY_DOUBLE_MATH_H
#define PHPSCI_NDARRAY_DOUBLE_MATH_H

#ifndef _MSC_VER
#include "../../config.h"
#endif
#include "../ndarray.h"

/* `float_abs` / `float_sqrt` are legacy float-precision scalar helpers that
   are currently unreferenced: every element-wise math method now dispatches
   through the typed paths — trig / hyperbolic / angle / rounding (incl.
   `round` with precision support) via `NDArray_TypedUnaryOp`, `arctan2` via
   the typed binary dispatch (`NDArray_Arctan2_*` / `cuda_atan2_*`). They are
   retained only so this translation unit stays in the build; drop them
   together with double_math.{c,h} in a future build-system cleanup. */
float float_abs(float val);
float float_sqrt(float val);
#endif //PHPSCI_NDARRAY_DOUBLE_MATH_H
