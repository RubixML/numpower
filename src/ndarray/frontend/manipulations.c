// HAVE_CUBLAS
#ifndef _MSC_VER
#include "../../../config.h"
#endif

#include <Zend/zend_interfaces.h>

// NDARRAY_TYPE_FLOAT32, NDARRAY_TYPE_FLOAT64
#include "../../types.h"

// NDArray, NDArray_TYPE
#include "../../ndarray.h"

// cuda_fill_float
#include "../../ndmath/cuda/cuda_math.h"

#include "manipulations.h"

// PRIVATE

void _fillFloat(NDArray *a, float fill_value) {
    int i;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_fill_float(NDArray_F32DATA(a), fill_value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            NDArray_F32DATA(a)[i] = fill_value;
        }
    }
}

void _fillDouble(NDArray *a, double fill_value) {
    int i;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_fill_double(NDArray_F64DATA(a), fill_value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            NDArray_F64DATA(a)[i] = fill_value;
        }
    }
}

//PUBLIC

void NDArray_fillByZval(NDArray *ndarray, zval *value)
{
    if (ndarray == NULL) {
        zend_throw_error(NULL, "Invalid NDArray");
    }

    if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT32 && Z_TYPE_P(value) == IS_DOUBLE) {
        _fillFloat(ndarray, (float)Z_DVAL_P(value));
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT32 && Z_TYPE_P(value) == IS_LONG) {
        _fillFloat(ndarray, (float)Z_LVAL_P(value));
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT32 && Z_TYPE_P(value) == IS_TRUE) {
        _fillFloat(ndarray, (float)1.0);
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT32 && Z_TYPE_P(value) == IS_FALSE) {
        _fillFloat(ndarray, (float)0.0);
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT64 && Z_TYPE_P(value) == IS_DOUBLE) {
        _fillDouble(ndarray, Z_DVAL_P(value));
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT64 && Z_TYPE_P(value) == IS_LONG) {
        _fillDouble(ndarray, (double)Z_LVAL_P(value));
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT64 && Z_TYPE_P(value) == IS_TRUE) {
        _fillDouble(ndarray, (double)1.0);
    } else if (NDArray_TYPE(ndarray) == NDARRAY_TYPE_FLOAT64 && Z_TYPE_P(value) == IS_FALSE) {
        _fillDouble(ndarray, (double)0.0);
    } else {
        zend_throw_error(NULL, "Invalid NDArray datatype");
    }
}
