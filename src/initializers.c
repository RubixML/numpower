#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "initializers.h"
#include "ndarray.h"
#include "types.h"
#include "../config.h"
#include "Zend/zend_hash.h"
#include "iterators.h"
#include "indexing.h"
#include <math.h>
#include <time.h>

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include "ndmath/cuda/cuda_math.h"
#include <curand.h>
#include <cublas_v2.h>
#include "ndmath/cuda/cuda_math.h"
#include "gpu_alloc.h"
#endif

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
#pragma ide diagnostic ignored "NullDereference"
/**
 *
 * @param arr
 * @param shape
 * @param ndim
 */
void get_zend_array_shape(zend_array* arr, int* shape, int ndim) {
    int i;

    if (shape == NULL && ndim != 0) {
        return;
    }

    if (zend_array_count(arr) == 0) {
        return;
    }

    // Initialize shape array to zeros
    for (i = 0; i < ndim; i++) {
        shape[i] = 0;
    }

    // Traverse the array to get the shape
    zval* val;
    ZEND_HASH_FOREACH_VAL(arr, val) {
        if (Z_TYPE_P(val) == IS_ARRAY) {
            get_zend_array_shape(Z_ARRVAL_P(val), shape + 1, ndim - 1);
            shape[0]++;
        } else {
            shape[0]++;
        }
    }
    ZEND_HASH_FOREACH_END();
}
#pragma clang diagnostic pop

/**
 *
 * @param arr
 * @return
 */
int is_packed_zend_array(zend_array *arr) {
    if (arr->nNumUsed == arr->nNumOfElements) {
        return 1;  // the array is packed
    } else {
        return 0;  // the array is sparse
    }
}

/**
 * Get number of dimensions from php_array
 *
 * @param arr
 * @return
 */
int get_num_dims_from_zval(zval *arr) {
    int num_dims = 0;

    if (zend_array_count(Z_ARRVAL_P(arr)) == 0) {
        return 1;
    }

    zval *val = zend_hash_index_find(Z_ARRVAL_P(arr), 0);
    while (val && Z_TYPE_P(val) == IS_ARRAY) {
        num_dims++;
        val = zend_hash_index_find(Z_ARRVAL_P(val), 0);
    }
    return num_dims+1;
}

/**
 * Create a new NDArray Descriptor
 *
 * @param numElements
 * @param elSize
 * @param type
 * @return
 */
NDArrayDescriptor* Create_Descriptor(long numElements, int elsize, const char* type) {
    NDArrayDescriptor* ndArrayDescriptor = emalloc(sizeof(NDArrayDescriptor));
    ndArrayDescriptor->elsize = elsize;
    ndArrayDescriptor->numElements = numElements;
    ndArrayDescriptor->type = type;
    return ndArrayDescriptor;
}

/**
 * Generate Strides Vector
 *
 * @return
 */
int* Generate_Strides(const int* dimensions, int dimensions_size, int elsize) {
    if (dimensions_size == 0 || dimensions == NULL) {
        return NULL;
    }

    int i;
    int * target_stride;
    target_stride = safe_emalloc(dimensions_size, sizeof(int), 0);

    for(i = 0; i < dimensions_size; i++) {
        target_stride[i] = 0;
    }

    target_stride[dimensions_size-1] = elsize;

    for(i = dimensions_size-2; i >= 0; i--) {
        target_stride[i] = dimensions[i+1] * target_stride[i+1];
    }

    return target_stride;
}

/**
 *
 * @param buffer
 * @param numElements
 * @param elsize
 */
void
NDArray_CreateBuffer(NDArray* array, int numElements, int elsize) {
    array->data = emalloc(numElements * elsize);
}

int iteration = 0;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
/**
 * @param target_carray
 */
void
NDArray_CopyFromZendArray(NDArray* target, zend_array* target_zval, int * first_index) {
    zval * element;
    float * data_double;

    ZEND_HASH_FOREACH_VAL(target_zval, element) {
        ZVAL_DEREF(element);
        switch (Z_TYPE_P(element)) {
            case IS_ARRAY:
                NDArray_CopyFromZendArray(target, Z_ARRVAL_P(element), first_index);
                break;
            case IS_LONG:
                convert_to_long(element);
                data_double = NDArray_FDATA(target);
                data_double[*first_index] = (float) zval_get_long(element);
                *first_index = *first_index + 1;
                break;
            case IS_TRUE:
                convert_to_long(element);
                data_double = NDArray_FDATA(target);
                data_double[*first_index] = (float) 1.0;
                *first_index = *first_index + 1;
                break;
            case IS_FALSE:
                convert_to_long(element);
                data_double = NDArray_FDATA(target);
                data_double[*first_index] = (float) 0.0;
                *first_index = *first_index + 1;
                break;
            case IS_DOUBLE:
                convert_to_double(element);
                data_double = NDArray_FDATA(target);
                data_double[*first_index] = (float) zval_get_double(element);
                *first_index = *first_index + 1;
                break;
            default:
                zend_throw_error(NULL, "an element with an invalid type was used at initialization");
                return;
        }
    }
    ZEND_HASH_FOREACH_END();
}
#pragma clang diagnostic pop

/**
 * Create NDArray from zend_array
 *
 * @param ht
 * @param ndim
 * @return
 */
NDArray* Create_NDArray_FromZendArray(zend_array* ht, int ndim) {
    int last_index = 0;
    int *shape;
    if (ndim != 0) {
        shape = ecalloc(ndim, sizeof(int));
    } else {
        shape = ecalloc(1, sizeof(int));
    }
    if (!is_packed_zend_array(ht)) {
        return NULL;
    }
    get_zend_array_shape(ht, shape, ndim);
    int total_num_elements = shape[0];

    // Calculate number of elements
    for (int i = 1; i < ndim; i++) {
        total_num_elements = total_num_elements * shape[i];
    }
    NDArray* array = Create_NDArray(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    if (ndim != 0) {
        NDArray_CreateBuffer(array, total_num_elements, get_type_size(NDARRAY_TYPE_FLOAT32));
        NDArray_CopyFromZendArray(array, ht, &last_index);
    } else {
        array->data = NULL;
        array->descriptor->numElements = 0;
    }
    return array;
}

/**
 * Create NDArray from PHP Object (zval)
 *
 * @param php_object
 * @return
 */
NDArray* Create_NDArray_FromZval(zval* php_object) {
    NDArray* new_array = NULL;
    if (Z_TYPE_P(php_object) == IS_ARRAY) {
        new_array = Create_NDArray_FromZendArray(Z_ARRVAL_P(php_object), get_num_dims_from_zval(php_object));
    }
    return new_array;
}

/**
 * Create basic NDArray from shape and type
 *
 * @param shape
 * @return
 */
NDArray*
Create_NDArray(int* shape, int ndim, const char* type, const int device) {
    NDArray* rtn;
    int type_size = get_type_size(type);

    if (shape == NULL) {
        return NULL;
    }

    long total_num_elements = shape[0];

    if (ndim == 0) {
        total_num_elements = 1;
    }

    // Calculate number of elements
    for (int i = 1; i < ndim; i++) {
        total_num_elements = total_num_elements * shape[i];
    }

    rtn = emalloc(sizeof(NDArray));
    rtn->descriptor = Create_Descriptor(total_num_elements, type_size, type);
    rtn->flags = 0;
    rtn->ndim = ndim;
    rtn->dimensions = shape;
    rtn->refcount = 1;
    rtn->base = NULL;
    rtn->device = device;
    rtn->strides = Generate_Strides(shape, ndim, type_size);
    NDArrayIterator_INIT(rtn);
    return rtn;
}

/**
 * Create an NDArray View from another NDArray with
 * a custom data pointer
 *
 * @param target
 * @param buffer_offset
 * @param shape
 * @param strides
 * @param ndim
 * @return
 */
NDArray*
NDArray_FromNDArrayBase(NDArray *target, char *data_ptr, int* shape, int* strides, const int ndim) {
    NDArray* rtn = emalloc(sizeof(NDArray));
    int total_num_elements = 1;

    rtn->strides = strides;
    rtn->dimensions = shape;

    for (int i = 0; i < ndim; i++) {
        total_num_elements = total_num_elements * NDArray_SHAPE(rtn)[i];
    }

    rtn->flags = 0;
    rtn->data = data_ptr;
    rtn->base = target;
    rtn->ndim = ndim;
    rtn->refcount = 1;
    rtn->device = NDArray_DEVICE(target);
    rtn->descriptor = Create_Descriptor(total_num_elements, sizeof(float), NDARRAY_TYPE_FLOAT32);
    NDArrayIterator_INIT(rtn);
    NDArray_ADDREF(target);
    return rtn;
}

/**
 * Create an NDArray View from another NDArray
 *
 * @param target
 * @param buffer_offset
 * @param shape
 * @param strides
 * @param ndim
 * @return
 */
NDArray*
NDArray_FromNDArray(NDArray *target, int buffer_offset, int* shape, int* strides, const int* ndim) {
    NDArray* rtn = emalloc(sizeof(NDArray));
    int total_num_elements = 1;
    int out_ndim;

    if (strides == NULL) {
        rtn->strides = emalloc(sizeof(int) * NDArray_NDIM(target));
        memcpy(NDArray_STRIDES(rtn), NDArray_STRIDES(target), sizeof(int) * NDArray_NDIM(target));
    }
    if (shape == NULL) {
        out_ndim = NDArray_NDIM(target);
        rtn->dimensions = emalloc(sizeof(int) * NDArray_NDIM(target));
        memcpy(NDArray_SHAPE(rtn), NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));
    }

    if (shape != NULL) {
        rtn->dimensions = shape;
        rtn->strides = strides;
        out_ndim = *ndim;
    }

    // Calculate number of elements
    for (int i = 0; i < out_ndim; i++) {
        total_num_elements = total_num_elements * NDArray_SHAPE(rtn)[i];
    }

    rtn->flags = 0;
    rtn->data = target->data + buffer_offset;
    rtn->base = target;
    rtn->ndim = out_ndim;
    rtn->refcount = 1;
    rtn->device = NDArray_DEVICE(target);
    rtn->descriptor = Create_Descriptor(total_num_elements, sizeof(float), NDARRAY_TYPE_FLOAT32);
    NDArrayIterator_INIT(rtn);
    NDArray_ADDREF(target);
    return rtn;
}

/**
 * Initialize NDArray with empty values
 *
 * @param shape
 * @param ndim
 * @return
 */
NDArray*
NDArray_Empty(int *shape, int ndim, const char *type, int device) {
    NDArray* rtn = Create_NDArray(shape, ndim, type, NDARRAY_DEVICE_CPU);

    if (rtn == NULL) {
        return rtn;
    }

    if (is_type(type, NDARRAY_TYPE_FLOAT32)) {
        if (device == NDARRAY_DEVICE_CPU) {
            rtn->device = NDARRAY_DEVICE_CPU;
            rtn->data = emalloc(NDArray_NUMELEMENTS(rtn) * sizeof(float));
        } else {
#ifdef HAVE_CUBLAS
            rtn->device = NDARRAY_DEVICE_GPU;
            vmalloc((void **) &rtn->data, NDArray_NUMELEMENTS(rtn) * sizeof(float));
#endif
        }
    }
    return rtn;
}

/**
 * @param a
 * @return
 */
NDArray*
NDArray_EmptyLike(NDArray *a) {
    int *output_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    memcpy(output_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
    return NDArray_Empty(output_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
}

/**
 * Initialize NDArray with zeros
 *
 * @param shape
 * @param ndim
 * @return
 */
NDArray*
NDArray_Zeros(int *shape, int ndim, const char *type, const int device) {
    NDArray* rtn = Create_NDArray(shape, ndim, type, device);

    if (rtn == NULL) {
        return rtn;
    }

    if (device == NDARRAY_DEVICE_CPU) {
        if (is_type(type, NDARRAY_TYPE_DOUBLE64)) {
            rtn->data = ecalloc(rtn->descriptor->numElements, sizeof(double));
        }
        if (is_type(type, NDARRAY_TYPE_FLOAT32)) {
            rtn->data = ecalloc(rtn->descriptor->numElements, sizeof(float));
        }
    }
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        if (is_type(type, NDARRAY_TYPE_DOUBLE64)) {
            vmalloc((void**)(&rtn->data), rtn->descriptor->numElements * sizeof(double));
            cudaMemset(rtn->data, 0, rtn->descriptor->numElements * sizeof(double));
        }
        if (is_type(type, NDARRAY_TYPE_FLOAT32)) {
            vmalloc((void**)(&rtn->data), rtn->descriptor->numElements * sizeof(float));
            cudaMemset(rtn->data, 0, rtn->descriptor->numElements * sizeof(float));
        }
    }
#endif
    return rtn;
}

/**
 * Initialize NDArray with zeros
 *
 * @param shape
 * @param ndim
 * @return
 */
NDArray*
NDArray_Ones(int *shape, int ndim, const char *type) {
    NDArray* rtn = Create_NDArray(shape, ndim, type, NDARRAY_DEVICE_CPU);
    if (rtn == NULL) {
        return NULL;
    }

    long i;
    rtn->data = emalloc(sizeof(float) * NDArray_NUMELEMENTS(rtn));
    for (i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
        NDArray_FDATA(rtn)[i] = (float)1.0;
    }
    return rtn;
}

/**
 * Identity Matrix
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Identity(int size) {
    NDArray *rtn;
    unsigned long index;
    int *shape;
    float *buffer_ptr;

    if (size < 0) {
        zend_throw_error(NULL, "negative dimensions are not allowed");
        return NULL;
    }

    if (size == 0) {
        shape = emalloc(sizeof(int) * 1);
        shape[0] = 0;
        return NDArray_Empty(shape, 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    }

    shape = emalloc(sizeof(int) * 2);
    shape[0] = size;
    shape[1] = size;
    rtn = NDArray_Zeros(shape, 2, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);

    buffer_ptr = NDArray_FDATA(rtn);
    // Set the diagonal elements to one with the specified stride
    for (int i = 0; i < size; i++) {
        index = ((i * size * sizeof(float)) + (i * sizeof(float))) / sizeof(float);
        buffer_ptr[(int)index] = 1.0f;
    }
    return rtn;
}

/**
 * Random samples from a truncated Gaussian distribution.
 *
 * The values generated are similar to values from a Normal distribution,
 * except that values more than two standard deviations from the mean are
 * discarded and re-drawn.
 * 
 * @param size
 * @return
 */
NDArray*
NDArray_TruncatedNormal(double loc, double scale, int* shape, int ndim, int accelerator) {
    NDArray *rtn;
    scale = scale / 0.88;

    if (accelerator == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
        int size = NDArray_NUMELEMENTS(rtn);
        cuda_truncated_normal(NDArray_FDATA(rtn), size, loc, scale);
#endif
    } else {
        rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
        for (int i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
            float z;
            do {
                float u1 = (float)rand() / (float)RAND_MAX;
                float u2 = (float)rand() / (float)RAND_MAX;
                z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
                z = (float)loc + (float)scale * z;
                NDArray_FDATA(rtn)[i] = z;
            } while (z < (loc - 2.0 * scale) || z > (loc + 2.0 * scale));
        }
    }
    return rtn;
}

/**
 * Random samples from a Gaussian distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Normal(double loc, double scale, int* shape, int ndim, int accelerator) {
    NDArray *rtn;
    if (accelerator == 1) {
#ifdef HAVE_CUBLAS
        rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
        if (rtn == NULL) {
            return NULL;
        }

        int size = NDArray_NUMELEMENTS(rtn);
        float* d_matrix = NULL;
    
        vmalloc((void**)&d_matrix, size * sizeof(float));
    
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateNormal(gen, d_matrix, size, (float)loc, (float)scale);
        
        if (rtn->data != NULL) {
            vfree(rtn->data);
        }
    
        rtn->data = (void*)d_matrix;
        curandDestroyGenerator(gen);
#endif
    } else {
        rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
        if (rtn == NULL) {
            return NULL;
        }

        int size = NDArray_NUMELEMENTS(rtn);

        // Generate random samples from the normal distribution
        for (int i = 0; i < size; i += 2) {
            float s, u, v;
            do {
                u = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0;
                v = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0;
                s = u * u + v * v;
            } while (s >= 1.0f || s ==0);

            float factor = sqrt(-2.0f * log(s) / s);
            float z1 = u * factor;

            NDArray_FDATA(rtn)[i] = (float)loc + (float)scale * z1;
            if (i + 1 < size) {
                float z2 = v * factor;
                NDArray_FDATA(rtn)[i+1] = (float)loc + (float)scale * z2;
            }
        }
    }
    
    return rtn;
}

/**
 * Random samples from a Gaussian distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_StandardNormal(int* shape, int ndim) {
    return NDArray_Normal(0.0, 1.0, shape, ndim, 0);
}

/**
 * Random samples from a Poisson distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Poisson(double lam, int* shape, int ndim) {
    NDArray *rtn;
    rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);

    // Generate random samples from the Poisson distribution
    for (int i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
        float L = expf((float)-lam);
        float p = 1.0f;
        int k = 0;

        do {
            k++;
            float u = (float)rand() / (float)RAND_MAX;
            p *= u;
        } while (p > L);
        NDArray_FDATA(rtn)[i] = (float)k - 1.0f;
    }

    return rtn;
}

/**
 * Random samples from a Poisson distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Uniform(double low, double high, int* shape, int ndim) {
    NDArray *rtn;
    rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    // Generate random samples from the normal distribution
    for (int i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
        float u = (float)rand() / (float)RAND_MAX;
        NDArray_FDATA(rtn)[i] = (float)low + u * ((float)high - (float)low);
    }
    return rtn;
}

/**
 * @param a
 * @return
 */
NDArray*
NDArray_Diag(NDArray *a) {
    int i;
    int index;
    NDArray *rtn;
    if (NDArray_NDIM(a) != 1 && NDArray_NDIM(a) != 2) {
        zend_throw_error(NULL, "Input array must be a vector or 2-dimensional");
        return NULL;
    }

    if (NDArray_NDIM(a) == 1) {
        int *rtn_shape = emalloc(sizeof(int) * 2);
        rtn_shape[0] = NDArray_NUMELEMENTS(a);
        rtn_shape[1] = NDArray_NUMELEMENTS(a);
        rtn = NDArray_Zeros(rtn_shape, 2, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);

        for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            index = ((i * NDArray_STRIDES(rtn)[0]) + (i * NDArray_STRIDES(rtn)[1])) / NDArray_ELSIZE(rtn);
            NDArray_FDATA(rtn)[index] = NDArray_FDATA(a)[i];
        }
    }

    if (NDArray_NDIM(a) == 2) {
        rtn = NDArray_Diagonal(a, 0);
        rtn->ndim = 1;
        rtn->dimensions[0] = NDArray_NUMELEMENTS(rtn);
        rtn->strides[0] = NDArray_ELSIZE(rtn);
    }
    return rtn;
}

/**
 * Fill values in place
 *
 * @param a
 * @return
 */
NDArray*
NDArray_Fill(NDArray *a, float fill_value) {
    int i;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_fill_float(NDArray_FDATA(a), fill_value, NDArray_NUMELEMENTS(a));
        return a;
#endif
    } else {
        for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            NDArray_FDATA(a)[i] = fill_value;
        }
    }
    return a;
}

/**
 * @param a
 * @return
 */
NDArray*
NDArray_Full(int *shape, int ndim,  double fill_value) {
    int *new_shape = emalloc(sizeof(int) * ndim);
    memcpy(new_shape, shape, sizeof(int) * ndim);
    NDArray *rtn = NDArray_Zeros(new_shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    return NDArray_Fill(rtn, (float)fill_value);
}

/**
 * Create NDArray from double
 * @return
 */
NDArray*
NDArray_CreateFromDoubleScalar(double scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(float);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT32;
    rtn->data = emalloc(sizeof(float));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((float*)rtn->data)[0] = (float)scalar;

    return rtn;
}

/**
 * Create NDArray from double
 * @return
 */
NDArray*
NDArray_CreateFromFloatScalar(float scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(float );
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT32;
    rtn->data = emalloc(sizeof(float));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((float *)rtn->data)[0] = scalar;

    return rtn;
}

/**
 * Create NDArray from long
 * @return
 */
NDArray*
NDArray_CreateFromLongScalar(long scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(float);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT32;
    rtn->data = emalloc(sizeof(float));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((float*)rtn->data)[0] = (float)scalar;

    return rtn;
}

/**
 * Copy NDArray
 *
 * @return
 */
NDArray*
NDArray_Copy(NDArray *a, int device) {
    NDArray *rtn;
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        rtn = emalloc(sizeof(NDArray));
        rtn->dimensions = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(rtn->dimensions, NDArray_SHAPE(a), NDArray_NDIM(a) * sizeof(int));
        rtn->strides = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(rtn->strides, NDArray_STRIDES(a), NDArray_NDIM(a) * sizeof(int));
        rtn->device = NDARRAY_DEVICE_GPU;
        rtn->refcount = 1;
        rtn->flags = 0;
        rtn->base = NULL;
        rtn->ndim = NDArray_NDIM(a);
        vmalloc((void **) &rtn->data, NDArray_NUMELEMENTS(a) * sizeof(float));
        cudaMemcpy(NDArray_FDATA(rtn), NDArray_FDATA(a), NDArray_NUMELEMENTS(a) * sizeof(float), cudaMemcpyDeviceToDevice);
        rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
        rtn->descriptor->numElements = NDArray_NUMELEMENTS(a);
        rtn->descriptor->elsize = NDArray_ELSIZE(a);
        rtn->descriptor->type = NDArray_TYPE(a);
        NDArrayIterator_INIT(rtn);
        return rtn;
#else
        return NULL;
#endif
    } else {
        rtn = emalloc(sizeof(NDArray));
        if (NDArray_NDIM(a) > 0) {
            rtn->dimensions = (int*)emalloc(sizeof(int) * NDArray_NDIM(a));
            memcpy(rtn->dimensions, NDArray_SHAPE(a), NDArray_NDIM(a) * sizeof(int));
            rtn->strides = (int*)emalloc(sizeof(int) * NDArray_NDIM(a));
            memcpy(rtn->strides, NDArray_STRIDES(a), NDArray_NDIM(a) * sizeof(int));
        } else {
            rtn->dimensions = emalloc(sizeof(int));
            rtn->strides = emalloc(sizeof(int));
        }
        rtn->device = NDARRAY_DEVICE_CPU;
        rtn->refcount = 1;
        rtn->flags = 0;
        rtn->ndim = NDArray_NDIM(a);
        rtn->base = NULL;
        rtn->data = emalloc(NDArray_NUMELEMENTS(a) * sizeof(float));
        memcpy(NDArray_DATA(rtn), NDArray_DATA(a), NDArray_NUMELEMENTS(a) * sizeof(float));
        rtn->descriptor = Create_Descriptor(NDArray_NUMELEMENTS(a), NDArray_ELSIZE(a), NDArray_TYPE(a));
        NDArrayIterator_INIT(rtn);
        return rtn;
    }
}

/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure
 */
static int safe_ceil_to_int(double value, int* ret) {
    double ivalue;

    ivalue = ceil(value);
    if (ivalue < INT_MIN || ivalue > INT_MAX) {
        return -1;
    }

    *ret = (int)ivalue;
    return 0;
}

/**
 * NDArray::arange
 *
 * @param start
 * @param stop
 * @param step
 * @return
 */
NDArray*
NDArray_Arange(double start, double stop, double step) {
    NDArray *rtn;
    int i;
    int length;

    if (safe_ceil_to_int((stop - start) / step, &length)) {
        zend_throw_error(NULL, "arange: overflow while computing length");
        return NULL;
    }

    if (length <= 0) {
        zend_throw_error(NULL, "arange: zero length");
        return NULL;
    }

    int *rtn_shape = emalloc(sizeof(int));
    rtn_shape[0] = length;
    rtn = NDArray_Zeros(rtn_shape, 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    NDArray_FDATA(rtn)[0] = (float)start;
    for (i = 1; i < length; i++) {
        NDArray_FDATA(rtn)[i] = NDArray_FDATA(rtn)[i-1] + step;
    }
    return rtn;
}

NDArray*
NDArray_Binomial(int *shape, int ndim, int n, float p) {
    // Calculate the total number of elements in the output array
    int total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    NDArray *rtn = NDArray_Zeros(shape, ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    // Generate random binomial numbers
    for (int i = 0; i < total_elements; i++) {
        int successes = 0;
        for (int j = 0; j < n; j++) {
            // Generate a random number between 0 and 1
            float random_value = (float)rand() / (float)RAND_MAX;
            if (random_value < p) {
                successes++;
            }
        }
        NDArray_FDATA(rtn)[i] = (float)successes;
    }
    return rtn;
}
