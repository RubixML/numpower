#ifndef PHPSCI_NDARRAY_DOUBLE_MATH_H
#define PHPSCI_NDARRAY_DOUBLE_MATH_H

#ifndef _MSC_VER
#include "../../config.h"
#endif
#include "../ndarray.h"

/* Legacy float-only helpers that survive the typed-unary refactor only
   because the corresponding PHP method is still on the legacy
   NDArray_Map path:
     - `float_arctan2` — binary op (NDArray_Map1ND).
     - `float_round`   — takes an extra precision argument (NDArray_Map1F).
   Both are out of scope for the unary dispatcher; refactoring them
   needs different infrastructure (binary-unary and precision-arg
   support, respectively).
   `float_abs` / `float_sqrt` are kept for ad-hoc usage elsewhere. */
float float_abs(float val);
float float_sqrt(float val);
float float_round(float number, float decimals);
float float_arctan2(float x, float y);
#endif //PHPSCI_NDARRAY_DOUBLE_MATH_H
