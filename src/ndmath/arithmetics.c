#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include <string.h>
#include <math.h>
#include "arithmetics.h"
#ifndef _MSC_VER
#include "../../config.h"
#endif
#include "../initializers.h"
#include "../iterators.h"
#include "../types.h"
#include "../manipulation.h"
#include "double_math.h"

#if HAVE_QUADMATH && NDARRAY_HAVE_FLOAT128
#  include <quadmath.h>
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda/cuda_math.h"
#include "../gpu_alloc.h"

#endif

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

#if HAVE_AVX2
#include <immintrin.h>
#endif

/**
 * Product of array element-wise
 *
 * @param a
 * @param b
 * @return
 */
float
NDArray_Float_Prod(NDArray* a) {
    float value = 1;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_prod_float(NDArray_NUMELEMENTS(a), NDArray_F32DATA(a), &value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value *= NDArray_F32DATA(a)[i];
        }
    }
    return value;
}

/**
 * Add elements of a element-wise
 *
 * @param a
 * @param b
 * @return
 */
float
NDArray_Sum_Float(NDArray* a) {
    float value = 0;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_sum_float(NDArray_NUMELEMENTS(a), NDArray_F32DATA(a), &value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value += NDArray_F32DATA(a)[i];
        }
    }
    return value;
}

/**
 * Add elements of a element-wise
 *
 * @param a
 * @param b
 * @return
 */
float
NDArray_Mean_Float(NDArray* a) {
    NDArray_Print(a, 0);
    float value = 0;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_sum_float(NDArray_NUMELEMENTS(a), NDArray_F32DATA(a), &value, NDArray_NUMELEMENTS(a));
        value = value / NDArray_NUMELEMENTS(a);
#endif
    } else {

#ifdef HAVE_CBLAS
        value = cblas_sasum(NDArray_NUMELEMENTS(a), NDArray_F32DATA(a), 1);
        value = value / NDArray_NUMELEMENTS(a);
#else
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value += NDArray_F32DATA(a)[i];
        }
        value = value / NDArray_NUMELEMENTS(a);
#endif
    }
    return value;
}

// Comparison function for sorting
int compare(const void* a, const void* b) {
    float fa = *((const float*)a);
    float fb = *((const float*)b);
    return (fa > fb) - (fa < fb);
}

float calculate_median(float* matrix, int size) {
    // Copy matrix elements to a separate array
    float* temp = malloc(size * sizeof(float));
    if (temp == NULL) {
        // Handle memory allocation error
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    memcpy(temp, matrix, size * sizeof(float));

    // Sort the array in ascending order
    qsort(temp, size, sizeof(float), compare);

    // Calculate the median value
    float median;
    if (size % 2 == 0) {
        // If the number of elements is even, average the two middle values
        median = (temp[size / 2 - 1] + temp[size / 2]) / 2.0f;
    } else {
        // If the number of elements is odd, take the middle value
        median = temp[size / 2];
    }

    // Free the temporary array
    free(temp);

    return median;
}

/**
 * Add elements of a element-wise
 *
 * @todo Implement GPU support
 * @param a
 * @param b
 * @return
 */
float
NDArray_Median_Float(NDArray* a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        zend_throw_error(NULL, "Median not available for GPU.");
        return -1;
#endif
    } else {
        return calculate_median(NDArray_F32DATA(a), NDArray_NUMELEMENTS(a));
    }
}

NDArray*
NDArray_Add_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillFloat(a, NDArray_F32DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillFloat(b, NDArray_F32DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise addition
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_add_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                            NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
#if HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            sub = _mm256_add_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], sub);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < numElements; i++) {
            resultData[i] = aData[i] + bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] + bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

NDArray* NDArray_Add_Double(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillDouble(a, NDArray_F64DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillDouble(b, NDArray_F64DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    if (!is_type(NDArray_TYPE(a), NDArray_TYPE(b))) {
        zend_throw_error(NULL, "Type mismatch, both NDArray MUST be the same type.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(double));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(double));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT64;
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise addition
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    double *resultData = (double *) result->data;
    double *aData = (double *) a_broad->data;
    double *bData = (double *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_add_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                            NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] + bData[i];
        }
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

__m256 fix_negative_zero(__m256 vec) {
    __m256 zero = _mm256_set1_ps(-0.0f);
    __m256 mask = _mm256_cmp_ps(vec, zero, _CMP_EQ_OQ);
    return _mm256_blendv_ps(vec, zero, mask);
}

/**
 * Multiply elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Multiply_Float(NDArray* a, NDArray* b) {
    NDArray *broadcasted = NULL;
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            int *shape = ecalloc(1, sizeof(int));
            NDArray *rtn = NDArray_Empty(shape, 0, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
            cuda_multiply_float(1, NDArray_F32DATA(a), NDArray_F32DATA(b), NDArray_F32DATA(rtn), 1);
            return rtn;
#endif
        } else {
            int *shape = ecalloc(1, sizeof(int));
            NDArray *rtn = NDArray_Empty(shape, 0, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
            NDArray_F32DATA(rtn)[0] = NDArray_F32DATA(a)[0] * NDArray_F32DATA(b)[0];
            return rtn;
        }
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillFloat(a, NDArray_F32DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillFloat(b, NDArray_F32DATA(b_temp)[0]);
    }

    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }
    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "NDArray_Multiply_Float: operands must be float32.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));

    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->device = NDArray_DEVICE(a_broad);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise product
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_multiply_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                            NDArray_NUMELEMENTS(a_broad));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
#if HAVE_AVX2
        int i = 0;
        __m256 vec1, vec2, mul;

        for (; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            mul = _mm256_mul_ps(vec1, vec2);
            mul = fix_negative_zero(mul); // Fix any -0.0 results
            _mm256_storeu_ps(&resultData[i], mul);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < numElements; i++) {
            resultData[i] = aData[i] * bData[i];
            if (resultData[i] == 0.0f && signbit(resultData[i])) {
                resultData[i] = 0.0f;
            }
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] * bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }

    return result;
}

/**
 * Subtract elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Subtract_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillFloat(a, NDArray_F32DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillFloat(b, NDArray_F32DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "NDArray_Subtract_Float: operands must be float32.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_subtract_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                            NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
#if HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            sub = _mm256_sub_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], sub);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < numElements; i++) {
            resultData[i] = aData[i] - bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] - bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

NDArray* NDArray_Subtract_Double(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillDouble(a, NDArray_F64DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillDouble(b, NDArray_F64DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    if (!is_type(NDArray_TYPE(a), NDArray_TYPE(b))) {
        zend_throw_error(NULL, "Types mismatch.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        if (is_type(NDArray_TYPE(a_broad), NDARRAY_TYPE_FLOAT32)) {
            vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        } else {
            vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(double));
        }
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        if (is_type(NDArray_TYPE(a_broad), NDARRAY_TYPE_FLOAT32)) {
            result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
        } else {
            result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(double));
        }
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDArray_TYPE(a_broad);
    result->descriptor->elsize = NDArray_ELSIZE(a_broad);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));

    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_subtract_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                            NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
        if (is_type(NDArray_TYPE(a_broad), NDARRAY_TYPE_FLOAT32)) {
            float *resultData = (float *) result->data;
            float *aData = (float *) a_broad->data;
            float *bData = (float *) b_broad->data;

#if HAVE_AVX2
            int i;
            __m256 vec1, vec2, sub;
    
            for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
                vec1 = _mm256_loadu_ps(&aData[i]);
                vec2 = _mm256_loadu_ps(&bData[i]);
                sub = _mm256_sub_ps(vec1, vec2);
                _mm256_storeu_ps(&resultData[i], sub);
            }
    
            // Handle remaining elements if the length is not a multiple of 4
            for (; i < numElements; i++) {
                resultData[i] = aData[i] - bData[i];
            }
#else
            for (int i = 0; i < numElements; i++) {
                resultData[i] = aData[i] - bData[i];
            }
#endif            
        } else {
            double *resultData = (double *) result->data;
            double *aData = (double *) a_broad->data;
            double *bData = (double *) b_broad->data;

            for (int i = 0; i < numElements; i++) {
                resultData[i] = aData[i] - bData[i];
            }
        }
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * Divide elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Divide_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));
        NDArray_F32DATA(rtn)[0] = NDArray_F32DATA(a)[0] / NDArray_F32DATA(b)[0];
        return rtn;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillFloat(a, NDArray_F32DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillFloat(b, NDArray_F32DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "NDArray_Divide_Float: operands must be float32.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->device = NDArray_DEVICE(a_broad);
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->device = NDArray_DEVICE(a_broad);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise division
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_divide_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                          NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
#if HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            sub = _mm256_div_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], sub);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < numElements; i++) {
            resultData[i] = aData[i] / bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] / bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Mod_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillFloat(a, NDArray_F32DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillFloat(b, NDArray_F32DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "NDArray_Mod_Float: operands must be float32.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_mod_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                       NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
#if HAVE_AVX2
        int i;
        __m256 vec1, vec2, vout;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            vout = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_floor_ps(_mm256_div_ps(vec1, vec2)), vec2));
            _mm256_storeu_ps(&resultData[i], vout);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < numElements; i++) {
            resultData[i] = fmodf(aData[i], bData[i]);
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = fmodf(aData[i], bData[i]);
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Pow_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillFloat(a, NDArray_F32DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillFloat(b, NDArray_F32DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "NDArray_Pow_Float: operands must be float32.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_pow_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                       NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = powf(aData[i], bData[i]);
        }
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

NDArray* NDArray_Pow_Double(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillDouble(a, NDArray_F64DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillDouble(b, NDArray_F64DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    if (!is_type(NDArray_TYPE(a), NDArray_TYPE(b))) {
        zend_throw_error(NULL, "Types mismatch.");
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        vmalloc((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(double));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(double));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT64;
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    double *resultData = (double *) result->data;
    double *aData = (double *) a_broad->data;
    double *bData = (double *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_pow_float(NDArray_NUMELEMENTS(a_broad), NDArray_F32DATA(a_broad), NDArray_F32DATA(b_broad), NDArray_F32DATA(result),
                       NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = pow(aData[i], bData[i]);
        }
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * NDArray::abs
 *
 * @param nda
 * @return
 */
NDArray*
NDArray_Abs(NDArray *nda) {
    NDArray *rtn = NULL;
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        if (NDArray_TYPE(nda) == NDARRAY_TYPE_FLOAT32) {
            rtn = NDArray_Map(nda, fabsf);

        } else {
            rtn = NDArray_Map_Double(nda, fabs);
        }
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_abs);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    return rtn;
}
//Doubles

double NDArray_Sum_Double(NDArray* a) {
    double value = 0;
    
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_sum_double(NDArray_NUMELEMENTS(a), NDArray_F64DATA(a), &value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value += NDArray_F64DATA(a)[i];
        }
    }
    return value;
}

NDArray *
NDArray_Multiply_Double(NDArray *a, NDArray *b)
{
    NDArray *broadcasted = NULL;
    NDArray *a_temp = NULL, *b_temp = NULL;

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Empty(shape, 0, NDARRAY_TYPE_FLOAT64, NDARRAY_DEVICE_CPU);
        NDArray_F64DATA(rtn)[0] = NDArray_F64DATA(a)[0] * NDArray_F64DATA(b)[0];
        return rtn;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillDouble(a, NDArray_F64DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillDouble(b, NDArray_F64DATA(b_temp)[0]);
    }

    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    result->device = NDARRAY_DEVICE_CPU;
    result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT64;
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));

    double *resultData = (double *) result->data;
    double *aData = (double *) a_broad->data;
    double *bData = (double *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);

    for (int i = 0; i < numElements; i++) {
        resultData[i] = aData[i] * bData[i];
    }

    if (a_temp != NULL) NDArray_FREE(a);
    if (b_temp != NULL) NDArray_FREE(b);
    if (broadcasted != NULL) NDArray_FREE(broadcasted);
    return result;
}

NDArray *
NDArray_Divide_Double(NDArray *a, NDArray *b)
{
    NDArray *a_temp = NULL, *b_temp = NULL;

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT64, NDARRAY_DEVICE_CPU);
        NDArray_F64DATA(rtn)[0] = NDArray_F64DATA(a)[0] / NDArray_F64DATA(b)[0];
        return rtn;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillDouble(a, NDArray_F64DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillDouble(b, NDArray_F64DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->device = NDARRAY_DEVICE_CPU;
    result->ndim = a_broad->ndim;
    result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT64;
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));

    double *resultData = (double *) result->data;
    double *aData = (double *) a_broad->data;
    double *bData = (double *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);

    for (int i = 0; i < numElements; i++) {
        resultData[i] = aData[i] / bData[i];
    }

    if (a_temp != NULL) NDArray_FREE(a);
    if (b_temp != NULL) NDArray_FREE(b);
    if (broadcasted != NULL) NDArray_FREE(broadcasted);
    return result;
}

NDArray *
NDArray_Mod_Double(NDArray *a, NDArray *b)
{
    NDArray *a_temp = NULL, *b_temp = NULL;

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b) && NDArray_NDIM(a) != 0 && NDArray_NDIM(b) != 0) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_FillDouble(a, NDArray_F64DATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_FillDouble(b, NDArray_F64DATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        if (a_temp != NULL) NDArray_FREE(a);
        if (b_temp != NULL) NDArray_FREE(b);
        if (broadcasted != NULL) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->device = NDARRAY_DEVICE_CPU;
    result->ndim = a_broad->ndim;
    result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT64;
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));

    double *resultData = (double *) result->data;
    double *aData = (double *) a_broad->data;
    double *bData = (double *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);

    for (int i = 0; i < numElements; i++) {
        resultData[i] = fmod(aData[i], bData[i]);
    }

    if (a_temp != NULL) NDArray_FREE(a);
    if (b_temp != NULL) NDArray_FREE(b);
    if (broadcasted != NULL) NDArray_FREE(broadcasted);
    return result;
}

/* ── float128 element-wise arithmetic helpers ─────────────────────────────── */

static NDArray *alloc_fp128_result(NDArray *a_broad) {
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    int ndim = NDArray_NDIM(a_broad) > 0 ? NDArray_NDIM(a_broad) : 1;
    result->strides    = (int *) emalloc(ndim * sizeof(int));
    result->dimensions = (int *) emalloc(ndim * sizeof(int));
    result->ndim       = NDArray_NDIM(a_broad);
    result->data       = (char *) emalloc((size_t)a_broad->descriptor->numElements * NDARRAY_FP128_SIZE);
    result->base       = NULL;
    result->flags      = 0;
    result->device     = NDARRAY_DEVICE_CPU;
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type        = NDARRAY_TYPE_FLOAT128;
    result->descriptor->elsize      = NDARRAY_FP128_SIZE;
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount   = 1;
    if (NDArray_NDIM(a_broad) > 0) {
        memcpy(result->strides,    a_broad->strides,    NDArray_NDIM(a_broad) * sizeof(int));
        memcpy(result->dimensions, a_broad->dimensions, NDArray_NDIM(a_broad) * sizeof(int));
    }
    NDArrayIterator_INIT(result);
    return result;
}

static NDArray *fp128_broadcast_scalar(NDArray *scalar, NDArray *other) {
    int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(other));
    copy(NDArray_SHAPE(other), n_shape, NDArray_NDIM(other));
    NDArray *expanded = NDArray_Zeros(n_shape, NDArray_NDIM(other),
                                     NDARRAY_TYPE_FLOAT128, NDARRAY_DEVICE_CPU);
    ndarray_fp128_t val;
    memcpy(&val, scalar->data, NDARRAY_FP128_SIZE);
    return NDArray_FillFloat128(expanded, val);
}

#define DEFINE_FP128_BINOP(NAME, EXPR)                                              \
NDArray* NDArray_##NAME##_Float128(NDArray* a, NDArray* b) {                        \
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_CPU || NDArray_DEVICE(b) != NDARRAY_DEVICE_CPU) { \
        zend_throw_error(NULL, "float128 arithmetic is CPU-only.");                 \
        return NULL;                                                                \
    }                                                                               \
    NDArray *a_temp = NULL, *b_temp = NULL;                                         \
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {                              \
        a_temp = a; a = fp128_broadcast_scalar(a, b);                               \
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {                       \
        b_temp = b; b = fp128_broadcast_scalar(b, a);                               \
    }                                                                               \
    NDArray *broadcasted = NULL, *a_broad, *b_broad;                                \
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {                          \
        broadcasted = NDArray_Broadcast(a, b); a_broad = broadcasted; b_broad = b;  \
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {                   \
        broadcasted = NDArray_Broadcast(b, a); b_broad = broadcasted; a_broad = a;  \
    } else { a_broad = a; b_broad = b; }                                            \
    NDArray *result = alloc_fp128_result(a_broad);                                  \
    ndarray_fp128_t *rd = NDArray_F128DATA(result);                                 \
    ndarray_fp128_t *ad = NDArray_F128DATA(a_broad);                                \
    ndarray_fp128_t *bd = NDArray_F128DATA(b_broad);                                \
    long n = result->descriptor->numElements;                                       \
    for (long i = 0; i < n; i++) { ndarray_fp128_t x = ad[i], y = bd[i]; rd[i] = (EXPR); } \
    if (a_temp) NDArray_FREE(a);                                                    \
    if (b_temp) NDArray_FREE(b);                                                    \
    if (broadcasted) NDArray_FREE(broadcasted);                                     \
    return result;                                                                  \
}

DEFINE_FP128_BINOP(Add,       x + y)
DEFINE_FP128_BINOP(Subtract,  x - y)
DEFINE_FP128_BINOP(Multiply,  x * y)
DEFINE_FP128_BINOP(Divide,    x / y)

NDArray* NDArray_Pow_Float128(NDArray* a, NDArray* b) {
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_CPU || NDArray_DEVICE(b) != NDARRAY_DEVICE_CPU) {
        zend_throw_error(NULL, "float128 arithmetic is CPU-only.");
        return NULL;
    }
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a; a = fp128_broadcast_scalar(a, b);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b; b = fp128_broadcast_scalar(b, a);
    }
    NDArray *broadcasted = NULL, *a_broad, *b_broad;
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b); a_broad = broadcasted; b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a); b_broad = broadcasted; a_broad = a;
    } else { a_broad = a; b_broad = b; }
    NDArray *result = alloc_fp128_result(a_broad);
    ndarray_fp128_t *rd = NDArray_F128DATA(result);
    ndarray_fp128_t *ad = NDArray_F128DATA(a_broad);
    ndarray_fp128_t *bd = NDArray_F128DATA(b_broad);
    long n = result->descriptor->numElements;
    for (long i = 0; i < n; i++) {
#if HAVE_QUADMATH && NDARRAY_HAVE_FLOAT128
        rd[i] = powq(ad[i], bd[i]);
#else
        rd[i] = (ndarray_fp128_t)powl((long double)ad[i], (long double)bd[i]);
#endif
    }
    if (a_temp) NDArray_FREE(a);
    if (b_temp) NDArray_FREE(b);
    if (broadcasted) NDArray_FREE(broadcasted);
    return result;
}

NDArray* NDArray_Mod_Float128(NDArray* a, NDArray* b) {
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_CPU || NDArray_DEVICE(b) != NDARRAY_DEVICE_CPU) {
        zend_throw_error(NULL, "float128 arithmetic is CPU-only.");
        return NULL;
    }
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a; a = fp128_broadcast_scalar(a, b);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b; b = fp128_broadcast_scalar(b, a);
    }
    NDArray *broadcasted = NULL, *a_broad, *b_broad;
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b); a_broad = broadcasted; b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a); b_broad = broadcasted; a_broad = a;
    } else { a_broad = a; b_broad = b; }
    NDArray *result = alloc_fp128_result(a_broad);
    ndarray_fp128_t *rd = NDArray_F128DATA(result);
    ndarray_fp128_t *ad = NDArray_F128DATA(a_broad);
    ndarray_fp128_t *bd = NDArray_F128DATA(b_broad);
    long n = result->descriptor->numElements;
    for (long i = 0; i < n; i++) {
#if HAVE_QUADMATH && NDARRAY_HAVE_FLOAT128
        rd[i] = fmodq(ad[i], bd[i]);
#else
        /* Lossless fp128 fmod using only native __float128 operators:
           fmod(a, b) = a - trunc(a/b)*b. trunc on __float128 implemented via
           casts to integer parts when in range, else exponent manipulation. */
        ndarray_fp128_t x = ad[i], y = bd[i];
        if (y == (ndarray_fp128_t)0) { rd[i] = (ndarray_fp128_t)(0.0/0.0); continue; }
        ndarray_fp128_t q = x / y;
        ndarray_fp128_t q_trunc;
        ndarray_fp128_t abs_q = q < (ndarray_fp128_t)0 ? -q : q;
        if (abs_q < (ndarray_fp128_t)9.2233720368547758e18) {
            /* Fits in int64 — truncate via cast (exact for in-range values). */
            long long li = (long long)q;
            q_trunc = (ndarray_fp128_t)li;
        } else {
            /* Out of int64 range; q is already at integer scale (mantissa
               can't represent fractional part above 2^113), so q itself is
               its own truncation. Guard with subtraction trick: trunc(x) =
               x - x%1, where x%1 reduces via integer subtraction. We rely on
               the fact that any |q| > 2^113 has no fractional bits. */
            q_trunc = q;
        }
        rd[i] = x - q_trunc * y;
#endif
    }
    if (a_temp) NDArray_FREE(a);
    if (b_temp) NDArray_FREE(b);
    if (broadcasted) NDArray_FREE(broadcasted);
    return result;
}

/* ──────────────────────────────────────────────────────────────────────────
   GPU typed binary op dispatch — keeps GPU arrays on GPU for every supported
   dtype. Handles scalar(0-dim)+array broadcast natively by filling a typed
   GPU buffer with the scalar value, then dispatching to the same-dtype
   kernel. Both operands must be on GPU and of the same dtype.

   Storage notes:
   - float128 on GPU is stored as double-double (16 bytes per element, layout
     (hi, lo)). Conversion to/from __float128 happens at the ToGPU/ToCPU
     boundary in src/ndarray.c. Precision is ~106 bits vs CPU fp128's 113.
   - float4 / float8 are 1-byte values; we currently don't have direct GPU
     kernels for them — those still route through CPU via NDArray_AsType's
     CPU fallback in the dispatcher. All 10 native ints + float16/64 + the
     dd-encoded float128 are direct on GPU.
   ────────────────────────────────────────────────────────────────────────── */

#ifdef HAVE_CUBLAS
/* Broadcast a 0-dim GPU scalar to a new GPU buffer matching `other`'s shape
   and dtype. Scalar may be CPU or GPU; we read it host-side after a copy. */
static NDArray *gpu_broadcast_scalar(NDArray *scalar, NDArray *other) {
    int *n_shape = emalloc(sizeof(int) * (NDArray_NDIM(other) > 0 ? NDArray_NDIM(other) : 1));
    if (NDArray_NDIM(other) > 0) {
        memcpy(n_shape, NDArray_SHAPE(other), NDArray_NDIM(other) * sizeof(int));
    } else {
        n_shape[0] = 1;
    }
    NDArray *expanded = NDArray_Zeros(n_shape, NDArray_NDIM(other),
                                     NDArray_TYPE(other), NDARRAY_DEVICE_GPU);
    if (expanded == NULL) return NULL;
    int n = (int)NDArray_NUMELEMENTS(expanded);
    const char *dt = NDArray_TYPE(other);

    /* Read scalar value from CPU side. */
    NDArray *scalar_cpu = (NDArray_DEVICE(scalar) == NDARRAY_DEVICE_CPU)
        ? scalar : NDArray_ToCPU(scalar);
    if (scalar_cpu == NULL) { NDArray_FREE(expanded); return NULL; }

    /* For dd128, scalar_cpu is stored as __float128. Need to split. */
    if (!strcmp(dt, "float128")) {
        ndarray_fp128_t v;
        memcpy(&v, scalar_cpu->data, NDARRAY_FP128_SIZE);
        double hi = (double)v;
        double lo = (double)(v - (ndarray_fp128_t)hi);
        cuda_fill_dd((double *)NDArray_DATA(expanded), hi, lo, n);
        if (scalar_cpu != scalar) NDArray_FREE(scalar_cpu);
        return expanded;
    }

    double dv = ndarray_element_to_double(NDArray_TYPE(scalar_cpu),
                                          NDArray_DATA(scalar_cpu), 0);
    if (!strcmp(dt, "int8"))    cuda_fill_i8 ((int8_t  *)NDArray_DATA(expanded), (int8_t)dv,  n);
    else if (!strcmp(dt, "uint8"))   cuda_fill_u8 ((uint8_t *)NDArray_DATA(expanded), (uint8_t)dv, n);
    else if (!strcmp(dt, "int16"))   cuda_fill_i16((int16_t *)NDArray_DATA(expanded), (int16_t)dv, n);
    else if (!strcmp(dt, "uint16"))  cuda_fill_u16((uint16_t*)NDArray_DATA(expanded), (uint16_t)dv,n);
    else if (!strcmp(dt, "int32"))   cuda_fill_i32((int32_t *)NDArray_DATA(expanded), (int32_t)dv, n);
    else if (!strcmp(dt, "uint32"))  cuda_fill_u32((uint32_t*)NDArray_DATA(expanded), (uint32_t)dv,n);
    else if (!strcmp(dt, "int64"))   cuda_fill_i64((int64_t *)NDArray_DATA(expanded), (int64_t)dv, n);
    else if (!strcmp(dt, "uint64"))  cuda_fill_u64((uint64_t*)NDArray_DATA(expanded), (uint64_t)dv,n);
    else if (!strcmp(dt, "float32")) cuda_fill_float((float *)NDArray_DATA(expanded), (float)dv,   n);
    else if (!strcmp(dt, "float64")) cuda_fill_f64((double *)NDArray_DATA(expanded), dv,           n);
    else if (!strcmp(dt, "float16")) cuda_fill_f16((uint16_t*)NDArray_DATA(expanded), (float)dv,   n);
    else {
        if (scalar_cpu != scalar) NDArray_FREE(scalar_cpu);
        NDArray_FREE(expanded);
        zend_throw_error(NULL, "Unsupported dtype for GPU scalar fill: %s", dt);
        return NULL;
    }

    if (scalar_cpu != scalar) NDArray_FREE(scalar_cpu);
    return expanded;
}

/* Allocate a GPU result NDArray with shape/dtype matching `a_broad`. */
static NDArray *gpu_alloc_result(NDArray *a_broad) {
    int *n_shape = emalloc(sizeof(int) * (NDArray_NDIM(a_broad) > 0 ? NDArray_NDIM(a_broad) : 1));
    if (NDArray_NDIM(a_broad) > 0) {
        memcpy(n_shape, NDArray_SHAPE(a_broad), NDArray_NDIM(a_broad) * sizeof(int));
    } else {
        n_shape[0] = 1;
    }
    return NDArray_Zeros(n_shape, NDArray_NDIM(a_broad),
                         NDArray_TYPE(a_broad), NDARRAY_DEVICE_GPU);
}

NDArray *NDArray_TypedBinOp_GPU(int opcode, NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) != NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray_TypedBinOp_GPU: both operands must be on GPU.");
        return NULL;
    }
    if (strcmp(NDArray_TYPE(a), NDArray_TYPE(b)) != 0) {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_GPU: both operands must have the same dtype (got %s and %s).",
            NDArray_TYPE(a), NDArray_TYPE(b));
        return NULL;
    }

    /* Broadcast 0-dim scalars to the other operand's shape. */
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        a = gpu_broadcast_scalar(a, b);
        if (a == NULL) return NULL;
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        b = gpu_broadcast_scalar(b, a);
        if (b == NULL) return NULL;
    }

    /* If shapes differ, broadcast each operand to the union shape on GPU.
       The union shape is the broadcast of a and b under standard NumPy rules.
       After this step a_broad and b_broad both have the union shape. */
    NDArray *a_broadcasted = NULL, *b_broadcasted = NULL;
    NDArray *a_broad = a, *b_broad = b;
    if (NDArray_NUMELEMENTS(a) != NDArray_NUMELEMENTS(b)
        || !NDArray_ShapeCompare(a, b)) {
        /* Compute union shape: take max(a_dim, b_dim) at each position after
           right-aligning. */
        int ndim_max = NDArray_NDIM(a) > NDArray_NDIM(b)
                     ? NDArray_NDIM(a) : NDArray_NDIM(b);
        int *union_shape = (int *)emalloc(sizeof(int) * (ndim_max > 0 ? ndim_max : 1));
        long union_n = 1;
        for (int d = 0; d < ndim_max; d++) {
            int ad = (d - (ndim_max - NDArray_NDIM(a)) >= 0)
                ? NDArray_SHAPE(a)[d - (ndim_max - NDArray_NDIM(a))] : 1;
            int bd = (d - (ndim_max - NDArray_NDIM(b)) >= 0)
                ? NDArray_SHAPE(b)[d - (ndim_max - NDArray_NDIM(b))] : 1;
            if (ad != bd && ad != 1 && bd != 1) {
                efree(union_shape);
                if (a_temp) NDArray_FREE(a);
                if (b_temp) NDArray_FREE(b);
                zend_throw_error(NULL,
                    "Cannot broadcast shapes on GPU (dim %d: %d vs %d).", d, ad, bd);
                return NULL;
            }
            union_shape[d] = ad > bd ? ad : bd;
            union_n *= union_shape[d];
        }

        /* Helper: broadcast `src` (GPU) to `union_shape` of length ndim_max.
           Returns a new GPU NDArray. Index mapping is computed on CPU once
           per call and shipped to GPU as a small int buffer. */
#define GPU_BROADCAST_TO_UNION(src_arr, padded_strides, out_var)                    \
        do {                                                                        \
            int padded_shape[NDARRAY_MAX_DIMS];                                     \
            int strides_elems[NDARRAY_MAX_DIMS];                                    \
            int src_nd = NDArray_NDIM(src_arr);                                     \
            int pad = ndim_max - src_nd;                                            \
            for (int d = 0; d < pad; d++) {                                         \
                padded_shape[d]   = 1;                                              \
                strides_elems[d]  = 0;                                              \
            }                                                                       \
            long stride_acc = 1;                                                    \
            for (int d = src_nd - 1; d >= 0; d--) {                                 \
                int td = pad + d;                                                   \
                padded_shape[td]  = NDArray_SHAPE(src_arr)[d];                      \
                strides_elems[td] = (padded_shape[td] > 1) ? (int)stride_acc : 0;   \
                if (padded_shape[td] > 1) stride_acc *= padded_shape[td];           \
            }                                                                       \
            int *src_offsets = (int *)emalloc(sizeof(int) * (size_t)union_n);       \
            for (long i = 0; i < union_n; i++) {                                    \
                long rem = i;                                                       \
                long off = 0;                                                       \
                for (int d = ndim_max - 1; d >= 0; d--) {                           \
                    long coord = rem % union_shape[d];                              \
                    rem /= union_shape[d];                                          \
                    off += coord * strides_elems[d];                                \
                }                                                                   \
                src_offsets[i] = (int)off;                                          \
            }                                                                       \
            int *src_offsets_gpu = NULL;                                            \
            vmalloc((void **)&src_offsets_gpu, sizeof(int) * (size_t)union_n);      \
            cudaMemcpy(src_offsets_gpu, src_offsets,                                \
                       sizeof(int) * (size_t)union_n, cudaMemcpyHostToDevice);      \
            efree(src_offsets);                                                     \
            int *target_shape_copy = (int *)emalloc(sizeof(int) * ndim_max);        \
            memcpy(target_shape_copy, union_shape, sizeof(int) * ndim_max);         \
            (out_var) = NDArray_Zeros(target_shape_copy, ndim_max,                  \
                                      NDArray_TYPE(src_arr), NDARRAY_DEVICE_GPU);   \
            if ((out_var) != NULL) {                                                \
                cuda_broadcast((const char *)NDArray_DATA(src_arr),                 \
                               (char *)NDArray_DATA(out_var),                       \
                               src_offsets_gpu, (int)union_n,                       \
                               NDArray_ELSIZE(src_arr));                            \
            }                                                                       \
            vfree(src_offsets_gpu);                                                 \
            (void)padded_strides;                                                   \
        } while (0)

        if (NDArray_NUMELEMENTS(a) != union_n) {
            GPU_BROADCAST_TO_UNION(a, NULL, a_broadcasted);
            if (a_broadcasted == NULL) {
                efree(union_shape);
                if (a_temp) NDArray_FREE(a);
                if (b_temp) NDArray_FREE(b);
                return NULL;
            }
            a_broad = a_broadcasted;
        }
        if (NDArray_NUMELEMENTS(b) != union_n) {
            GPU_BROADCAST_TO_UNION(b, NULL, b_broadcasted);
            if (b_broadcasted == NULL) {
                efree(union_shape);
                if (a_broadcasted) NDArray_FREE(a_broadcasted);
                if (a_temp) NDArray_FREE(a);
                if (b_temp) NDArray_FREE(b);
                return NULL;
            }
            b_broad = b_broadcasted;
        }
        efree(union_shape);
#undef GPU_BROADCAST_TO_UNION
    }

    NDArray *result = gpu_alloc_result(a_broad);
    if (result == NULL) {
        if (a_temp) NDArray_FREE(a);
        if (b_temp) NDArray_FREE(b);
        return NULL;
    }
    int n = (int)NDArray_NUMELEMENTS(result);
    const char *dt = NDArray_TYPE(result);

#define DISPATCH_OP(DTSTR, TAG, T)                                                  \
    if (!strcmp(dt, DTSTR)) {                                                       \
        T *ap = (T *)NDArray_DATA(a_broad);                                         \
        T *bp = (T *)NDArray_DATA(b_broad);                                         \
        T *rp = (T *)NDArray_DATA(result);                                          \
        switch (opcode) {                                                           \
            case ZEND_ADD: cuda_add_##TAG(ap, bp, rp, n); break;                    \
            case ZEND_SUB: cuda_sub_##TAG(ap, bp, rp, n); break;                    \
            case ZEND_MUL: cuda_mul_##TAG(ap, bp, rp, n); break;                    \
            case ZEND_DIV: cuda_div_##TAG(ap, bp, rp, n); break;                    \
            case ZEND_MOD: cuda_mod_##TAG(ap, bp, rp, n); break;                    \
            case ZEND_POW: cuda_pow_##TAG(ap, bp, rp, n); break;                    \
            default:                                                                \
                zend_throw_error(NULL, "Unsupported opcode for GPU typed binop.");  \
                NDArray_FREE(result); result = NULL;                                \
        }                                                                           \
        goto done;                                                                  \
    }

    DISPATCH_OP("int8",    i8,  int8_t)
    DISPATCH_OP("uint8",   u8,  uint8_t)
    DISPATCH_OP("int16",   i16, int16_t)
    DISPATCH_OP("uint16",  u16, uint16_t)
    DISPATCH_OP("int32",   i32, int32_t)
    DISPATCH_OP("uint32",  u32, uint32_t)
    DISPATCH_OP("int64",   i64, int64_t)
    DISPATCH_OP("uint64",  u64, uint64_t)
    DISPATCH_OP("float64", f64, double)
    DISPATCH_OP("float16", f16, uint16_t)
#undef DISPATCH_OP
    /* float128 on GPU: stored as dd (2*n doubles). The kernels operate on
       the same byte layout. */
    if (!strcmp(dt, "float128")) {
        double *ap = (double *)NDArray_DATA(a_broad);
        double *bp = (double *)NDArray_DATA(b_broad);
        double *rp = (double *)NDArray_DATA(result);
        switch (opcode) {
            case ZEND_ADD: cuda_add_dd(ap, bp, rp, n); break;
            case ZEND_SUB: cuda_sub_dd(ap, bp, rp, n); break;
            case ZEND_MUL: cuda_mul_dd(ap, bp, rp, n); break;
            case ZEND_DIV: cuda_div_dd(ap, bp, rp, n); break;
            case ZEND_MOD: cuda_mod_dd(ap, bp, rp, n); break;
            case ZEND_POW: cuda_pow_dd(ap, bp, rp, n); break;
            default:
                zend_throw_error(NULL, "Unsupported opcode for GPU typed binop.");
                NDArray_FREE(result); result = NULL;
        }
        goto done;
    }
    /* float32: route to the existing kernel for backwards compat. */
    if (!strcmp(dt, "float32")) {
        float *ap = (float *)NDArray_DATA(a_broad);
        float *bp = (float *)NDArray_DATA(b_broad);
        float *rp = (float *)NDArray_DATA(result);
        switch (opcode) {
            case ZEND_ADD: cuda_add_float(n, ap, bp, rp, n); break;
            case ZEND_SUB: cuda_subtract_float(n, ap, bp, rp, n); break;
            case ZEND_MUL: cuda_multiply_float(n, ap, bp, rp, n); break;
            case ZEND_DIV: cuda_divide_float(n, ap, bp, rp, n); break;
            case ZEND_MOD: cuda_mod_float(n, ap, bp, rp, n); break;
            case ZEND_POW: cuda_pow_float(n, ap, bp, rp, n); break;
            default:
                zend_throw_error(NULL, "Unsupported opcode for GPU typed binop.");
                NDArray_FREE(result); result = NULL;
        }
        goto done;
    }

    /* float4 / float8: cast both operands to float16 on GPU, compute, cast
       back to the source dtype. Both formats fit losslessly in float16
       (fp4: 8 representable positive values; fp8 E4M3: max 240, mantissa 3).
       Conversion uses dedicated kernels in cuda_math.cu. */
    if (!strcmp(dt, "float4") || !strcmp(dt, "float8")) {
        /* Allocate float16 staging buffers on GPU. */
        int *shape_a16 = (int *)emalloc(sizeof(int) * (NDArray_NDIM(a_broad) > 0 ? NDArray_NDIM(a_broad) : 1));
        int *shape_b16 = (int *)emalloc(sizeof(int) * (NDArray_NDIM(a_broad) > 0 ? NDArray_NDIM(a_broad) : 1));
        if (NDArray_NDIM(a_broad) > 0) {
            memcpy(shape_a16, NDArray_SHAPE(a_broad), NDArray_NDIM(a_broad) * sizeof(int));
            memcpy(shape_b16, NDArray_SHAPE(a_broad), NDArray_NDIM(a_broad) * sizeof(int));
        } else {
            shape_a16[0] = 1; shape_b16[0] = 1;
        }
        NDArray *a16 = NDArray_Zeros(shape_a16, NDArray_NDIM(a_broad), "float16", NDARRAY_DEVICE_GPU);
        NDArray *b16 = NDArray_Zeros(shape_b16, NDArray_NDIM(a_broad), "float16", NDARRAY_DEVICE_GPU);
        if (a16 == NULL || b16 == NULL) {
            if (a16) NDArray_FREE(a16);
            if (b16) NDArray_FREE(b16);
            NDArray_FREE(result);
            result = NULL;
            goto done;
        }
        if (!strcmp(dt, "float4")) {
            cuda_cast_fp4_to_f16((uint8_t *)NDArray_DATA(a_broad), (uint16_t *)NDArray_DATA(a16), n);
            cuda_cast_fp4_to_f16((uint8_t *)NDArray_DATA(b_broad), (uint16_t *)NDArray_DATA(b16), n);
        } else {
            cuda_cast_fp8_to_f16((uint8_t *)NDArray_DATA(a_broad), (uint16_t *)NDArray_DATA(a16), n);
            cuda_cast_fp8_to_f16((uint8_t *)NDArray_DATA(b_broad), (uint16_t *)NDArray_DATA(b16), n);
        }
        /* Compute in float16. */
        NDArray *r16 = NDArray_TypedBinOp_GPU(opcode, a16, b16);
        NDArray_FREE(a16);
        NDArray_FREE(b16);
        if (r16 == NULL) {
            NDArray_FREE(result);
            result = NULL;
            goto done;
        }
        /* Cast float16 result back to fp4/fp8. */
        if (!strcmp(dt, "float4")) {
            cuda_cast_f16_to_fp4((uint16_t *)NDArray_DATA(r16), (uint8_t *)NDArray_DATA(result), n);
        } else {
            cuda_cast_f16_to_fp8((uint16_t *)NDArray_DATA(r16), (uint8_t *)NDArray_DATA(result), n);
        }
        NDArray_FREE(r16);
        goto done;
    }

    /* Unhandled dtype. Free result and signal failure. */
    NDArray_FREE(result);
    result = NULL;
    zend_throw_error(NULL, "GPU typed binop: unsupported dtype %s", dt);

done:
    if (a_temp) NDArray_FREE(a);
    if (b_temp) NDArray_FREE(b);
    if (a_broadcasted) NDArray_FREE(a_broadcasted);
    if (b_broadcasted) NDArray_FREE(b_broadcasted);
    return result;
}
#else  /* HAVE_CUBLAS */
NDArray *NDArray_TypedBinOp_GPU(int opcode, NDArray *a, NDArray *b) {
    (void)opcode; (void)a; (void)b;
    zend_throw_error(NULL, "GPU typed binop: extension built without CUDA support.");
    return NULL;
}
#endif
