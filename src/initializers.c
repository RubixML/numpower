#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "initializers.h"
#include "ndarray.h"
#include "ndarray/frontend/ndarray_factory.h"
#include "buffer.h"
#include "types.h"
#ifndef _MSC_VER
#include "../config.h"
#endif
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

/**
 * @brief Create a new NDArray descriptor.
 *
 * @param numElements element count this buffer will hold
 * @param elsize      per-element size in bytes
 * @param type        dtype tag (one of NDARRAY_TYPE_*)
 * @return            newly allocated NDArrayDescriptor (caller owns)
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
 * @brief Build a float32 NDArray from a PHP array zval (legacy entry).
 *
 * Used by `ZVAL_TO_NDARRAY` when an arithmetic op receives a raw PHP
 * array as one operand. Delegates to the modern factory so it benefits
 * from sparse-key handling and rectangular-shape validation; ragged
 * inputs now raise a catchable `Error` (was a fatal `E_ERROR` before).
 *
 * The modern factory registers every created NDArray in the global
 * buffer. The legacy `ZVAL_TO_NDARRAY` contract, however, is that the
 * temporary is owned by the immediate caller and released with a plain
 * `NDArray_FREE`; the caller never touches `buffer_ndarray_free`. To
 * keep that contract we detach the new NDArray from the buffer here so
 * `NDArray_FREE` is balanced and no stale slot survives past the call.
 *
 * Non-array zvals are not handled here — `ZVAL_TO_NDARRAY` routes
 * scalars and NDArray objects through dedicated factories.
 *
 * @param[in] php_object zval; must be IS_ARRAY for a non-NULL return
 * @return    new NDArray on success, NULL otherwise (caller frees with NDArray_FREE)
 */
NDArray* Create_NDArray_FromZval(zval* php_object) {
    if (Z_TYPE_P(php_object) != IS_ARRAY) {
        return NULL;
    }
    NDArray *rtn = NDArrayFactory_createFromZval(php_object, NDARRAY_TYPE_FLOAT32);
    if (rtn != NULL && rtn->uuid != -1) {
        buffer_replace(rtn->uuid, NULL);
        rtn->uuid = -1;
    }
    return rtn;
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
    rtn->uuid = -1;
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
    long total_num_elements = 1;

    rtn->strides = strides;
    rtn->dimensions = shape;

    for (int i = 0; i < ndim; i++) {
        total_num_elements = total_num_elements * (long)NDArray_SHAPE(rtn)[i];
    }
    if (ndim == 0) {
        total_num_elements = 1;
    }

    rtn->flags = 0;
    rtn->data = data_ptr;
    rtn->base = target;
    rtn->ndim = ndim;
    rtn->refcount = 1;
    rtn->device = NDArray_DEVICE(target);
    /* Preserve the source dtype/elsize. Hardcoding float32 here corrupted
       any non-float32 view (squeeze, slice, atleast1d/2d/3d, …) by re-
       interpreting the underlying bytes — see slice() across all dtypes. */
    rtn->descriptor = Create_Descriptor(total_num_elements,
                                        NDArray_ELSIZE(target),
                                        NDArray_TYPE(target));
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

    int elsize = get_type_size(type);
    if (elsize == 0) return NULL;

    if (device == NDARRAY_DEVICE_CPU) {
        rtn->device = NDARRAY_DEVICE_CPU;
        rtn->data = emalloc((size_t)NDArray_NUMELEMENTS(rtn) * (size_t)elsize);
    }
#ifdef HAVE_CUBLAS
    else {
        rtn->device = NDARRAY_DEVICE_GPU;
        vmalloc((void **)&rtn->data, (size_t)NDArray_NUMELEMENTS(rtn) * (size_t)elsize);
    }
#endif
    return rtn;
}

/**
 * @param a
 * @return
 */
NDArray* NDArray_EmptyLike(NDArray *a) {
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
    /* Validate the dtype before any allocation so a bogus string can't leak
       the descriptor/dimensions stored inside Create_NDArray on the `elsize
       == 0` early-return that used to follow. */
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) {
            efree(shape);
        }
        return NULL;
    }

    NDArray* rtn = Create_NDArray(shape, ndim, type, device);
    if (rtn == NULL) {
        return rtn;
    }

    size_t total = (size_t)rtn->descriptor->numElements * (size_t)elsize;

    if (device == NDARRAY_DEVICE_CPU) {
        /* ecalloc(0, n) is well-defined and returns a pointer that NDArray_FREE
           can release, so the empty-shape path stays balanced under VCHECK. */
        rtn->data = ecalloc((size_t)rtn->descriptor->numElements, (size_t)elsize);
    }
#ifdef HAVE_CUBLAS
    else if (device == NDARRAY_DEVICE_GPU) {
        /* Allocate the buffer directly in VRAM and zero it on-device — no
           host memory is touched, no H2D copy is needed. `vmalloc(0)` is a
           safe no-op that leaves data == NULL; `cudaMemset` then skips. */
        rtn->data = NULL;
        vmalloc((void **)(&rtn->data), (unsigned int) total);
        if (total > 0 && rtn->data != NULL) {
            cudaMemset(rtn->data, 0, total);
        }
    }
#else
    else if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before passing
           NDARRAY_DEVICE_GPU. If they didn't, fail loudly rather than hand
           back an NDArray with an uninitialised `data` pointer. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif
    return rtn;
}

/**
 * @brief Broadcast a host-encoded scalar across @p rtn's data buffer.
 *
 * Shared by every typed initializer that fills with a constant
 * (`NDArray_Ones`, `NDArray_Full`). The @p encoded buffer holds one
 * element in the dtype's host representation (16 bytes covers the widest
 * dtype, fp128). On CPU the buffer is broadcast via `memset` for
 * single-byte dtypes or a `memcpy` loop otherwise. On GPU the work goes
 * through `cuda_fill_bytes`, which performs the broadcast entirely on
 * device — only the seed @p encoded source crosses the PCIe bus.
 *
 * fp128 on GPU is special: the on-device storage is a (hi, lo) double-
 * double pair while @p encoded is in native __float128 layout. The helper
 * detects this and converts to DD via `ndarray_fp128_to_double` before
 * the broadcast. For values that don't fit in a single double the lo
 * word carries the rounding remainder.
 *
 * @param[in,out] rtn     NDArray to fill in place.
 * @param[in]     encoded ≥ NDArray_ELSIZE(rtn) bytes; first elsize bytes
 *                        are the source element in host representation.
 */
static void
ndarray_broadcast_encoded(NDArray *rtn, const char *encoded) {
    long n      = NDArray_NUMELEMENTS(rtn);
    int  elsize = NDArray_ELSIZE(rtn);
    int  device = NDArray_DEVICE(rtn);
    const char *type = NDArray_TYPE(rtn);

    if (n <= 0) {
        return;
    }

    if (device == NDARRAY_DEVICE_CPU) {
        char *data = (char *)rtn->data;
        if (elsize == 1) {
            memset(data, encoded[0], (size_t)n);
        } else {
            for (long i = 0; i < n; i++) {
                memcpy(data + (size_t)i * (size_t)elsize,
                       encoded, (size_t)elsize);
            }
        }
        return;
    }
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        if (!strcmp(type, "float128")) {
            /* Host fp128 → on-device DD (hi, lo) pair. For (lo == 0)
               specials (±INF, NaN, ±0) the +0.0 companion preserves the
               sign of zero per IEEE-754. */
            ndarray_fp128_t v;
            memcpy(&v, encoded, NDARRAY_FP128_SIZE);
#if NDARRAY_HAVE_FLOAT128
            double hi = (double)v;
            double lo;
            if (isinf(hi) || isnan(hi)) {
                lo = 0.0;
            } else {
                lo = (double)(v - (ndarray_fp128_t)hi);
            }
#else
            double hi = v.hi;
            double lo = (isinf(hi) || isnan(hi)) ? 0.0 : v.lo;
#endif
            double dd[2] = { hi, lo };
            cuda_fill_bytes((char *)rtn->data, (const char *)dd,
                            sizeof(dd), n);
        } else {
            cuda_fill_bytes((char *)rtn->data, encoded,
                            (size_t)elsize, n);
        }
    }
#endif
}

/**
 * @brief Allocate an NDArray of the requested shape / dtype / device and
 *        fill every element with the dtype-appropriate representation of 1.
 *
 * The previous implementation always allocated `numElements * sizeof(float)`
 * bytes and broadcast a float32 1.0 into them regardless of @p type — a
 * dormant buffer-size + wrong-value bug that this rewrite fixes by routing
 * through `NDArray_Empty` (dtype-aware allocator on both devices) and
 * encoding the scalar via `ndarray_set_from_double` once. The actual
 * broadcast is delegated to `ndarray_broadcast_encoded` so it stays in
 * sync with `NDArray_Full`.
 *
 * @param[in] shape  Newly-allocated int[ndim]; ownership transfers to the
 *                   returned NDArray (it lives in `rtn->dimensions`).
 * @param[in] ndim   Number of dimensions; 0 yields a 0-D scalar of value 1.
 * @param[in] type   Canonical dtype string (one of the 14 supported aliases).
 * @param[in] device Target device — NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray on success, NULL on validation / allocation failure.
 */
NDArray*
NDArray_Ones(int *shape, int ndim, const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) {
            efree(shape);
        }
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before requesting GPU.
           If they didn't, fail loudly instead of returning an NDArray with
           uninitialised on-device storage. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    /* Encode the dtype's representation of 1.0 once into a 16-byte stack
       scratch (16 = NDARRAY_FP128_SIZE, the widest dtype). All narrower
       dtypes use only the leading `elsize` bytes; the unused tail stays
       zero from the memset. */
    char encoded[NDARRAY_FP128_SIZE];
    memset(encoded, 0, sizeof(encoded));
    ndarray_set_from_double(type, encoded, 0, 1.0);

    ndarray_broadcast_encoded(rtn, encoded);
    return rtn;
}

/**
 * @brief Allocate an NDArray of the requested shape / dtype / device and
 *        fill every element with the pre-encoded scalar in @p encoded.
 *
 * Companion to `NDArray_Ones` and `NDArray_Zeros`. Callers encode the
 * dtype-appropriate fill value into @p encoded once
 * (see `NDArray_EncodeZvalToDtype`) and then this function delegates the
 * actual broadcast to `ndarray_broadcast_encoded`, which keeps the
 * CPU/GPU dispatch (incl. fp128 host-DD conversion on GPU) in one place.
 *
 * @param[in] shape   Newly-allocated int[ndim]; ownership transfers into
 *                    the returned NDArray's `dimensions`.
 * @param[in] ndim    Number of dimensions; 0 yields a 0-D scalar.
 * @param[in] type    Canonical dtype string.
 * @param[in] device  NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @param[in] encoded ≥ get_type_size(type) bytes carrying the host
 *                    representation of the scalar to broadcast.
 * @return New NDArray on success, NULL on validation / allocation failure.
 */
NDArray*
NDArray_Full(int *shape, int ndim, const char *type, int device,
             const char *encoded) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) {
            efree(shape);
        }
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    ndarray_broadcast_encoded(rtn, encoded);
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

    buffer_ptr = NDArray_F32DATA(rtn);
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
        cuda_truncated_normal(NDArray_F32DATA(rtn), size, loc, scale);
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
                NDArray_F32DATA(rtn)[i] = z;
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

            NDArray_F32DATA(rtn)[i] = (float)loc + (float)scale * z1;
            if (i + 1 < size) {
                float z2 = v * factor;
                NDArray_F32DATA(rtn)[i + 1] = (float)loc + (float)scale * z2;
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
        NDArray_F32DATA(rtn)[i] = (float)k - 1.0f;
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
        NDArray_F32DATA(rtn)[i] = (float)low + u * ((float)high - (float)low);
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
            NDArray_F32DATA(rtn)[index] = NDArray_F32DATA(a)[i];
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
NDArray_FillFloat(NDArray *a, float fill_value) {
    int i;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_fill_float(NDArray_F32DATA(a), fill_value, NDArray_NUMELEMENTS(a));
        return a;
#endif
    } else {
        for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            NDArray_F32DATA(a)[i] = fill_value;
        }
    }
    return a;
}

/**
 * @brief Broadcast @p fill_value across every element of a float64 NDArray @p a.
 *
 * Pre-existing bug fixed: the GPU branch used to dispatch to
 * `cuda_fill_float`, writing 4-byte float32 values into the 8-byte float64
 * buffer and silently corrupting the on-device data of every other element
 * (and trampling N/2 elements past the end of the requested range). The
 * GPU path now dispatches to `cuda_fill_double` to match the buffer dtype.
 *
 * @param[in,out] a          float64 NDArray to fill in place (CPU or GPU).
 * @param[in]     fill_value Scalar value broadcast across every element.
 * @return @p a (unchanged pointer, mutated in place).
 */
NDArray* NDArray_FillDouble(NDArray *a, double fill_value) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_fill_double(NDArray_F64DATA(a), fill_value,
                         NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            NDArray_F64DATA(a)[i] = fill_value;
        }
    }
    return a;
}

NDArray* NDArray_FillFloat128(NDArray *a, ndarray_fp128_t fill_value) {
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_CPU) {
        zend_throw_error(NULL, "NDArray_FillFloat128: float128 is CPU-only.");
        return NULL;
    }
    for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        NDArray_F128DATA(a)[i] = fill_value;
    }
    return a;
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
    rtn->descriptor->elsize = sizeof(float);
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

    rtn->uuid = -1;
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
    size_t nbytes = (size_t)NDArray_NUMELEMENTS(a) * (size_t)NDArray_ELSIZE(a);
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        rtn = emalloc(sizeof(NDArray));
        rtn->uuid = -1;
        rtn->dimensions = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(rtn->dimensions, NDArray_SHAPE(a), NDArray_NDIM(a) * sizeof(int));
        rtn->strides = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(rtn->strides, NDArray_STRIDES(a), NDArray_NDIM(a) * sizeof(int));
        rtn->device = NDARRAY_DEVICE_GPU;
        rtn->refcount = 1;
        rtn->flags = 0;
        rtn->base = NULL;
        rtn->ndim = NDArray_NDIM(a);
        vmalloc((void **) &rtn->data, nbytes);
        if (nbytes > 0) {
            cudaMemcpy(rtn->data, NDArray_DATA(a), nbytes, cudaMemcpyDeviceToDevice);
        }
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
        rtn->uuid = -1;
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
        rtn->data = emalloc(nbytes);
        memcpy(NDArray_DATA(rtn), NDArray_DATA(a), nbytes);
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
    NDArray_F32DATA(rtn)[0] = (float)start;
    for (i = 1; i < length; i++) {
        NDArray_F32DATA(rtn)[i] = NDArray_F32DATA(rtn)[i - 1] + step;
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
        NDArray_F32DATA(rtn)[i] = (float)successes;
    }
    return rtn;
}

NDArray* NDArrayFactory_CreateFromDoubleScalar(double scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(double);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT64;
    rtn->data = emalloc(sizeof(double));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((double*)rtn->data)[0] = (double)scalar;
    return rtn;
}
