#ifndef PHPSCI_NDARRAY_DOUBLE_MATH_H
#define PHPSCI_NDARRAY_DOUBLE_MATH_H

#ifndef _MSC_VER
#include "../../config.h"
#endif
#include "../ndarray.h"

/* Legacy float-only helpers that survive the typed-unary refactor only
   because the corresponding PHP method is still on the legacy
   NDArray_Map path:
     - `float_round` — takes an extra precision argument (NDArray_Map1F).
   It is out of scope for the unary dispatcher; refactoring it needs
   precision-arg support. `arctan2` moved to the typed binary dispatch
   (`NDArray_Arctan2_*` / `cuda_atan2_*`).
   `float_abs` / `float_sqrt` are kept for ad-hoc usage elsewhere. */
float float_abs(float val);
float float_sqrt(float val);
float float_round(float number, float decimals);
#endif //PHPSCI_NDARRAY_DOUBLE_MATH_H
