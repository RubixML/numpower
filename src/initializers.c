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
#include <limits.h>
#include <math.h>
#include <stdint.h>
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
 * @brief Allocate an NDArray with uninitialised storage of @p ndim × @p shape.
 *
 * Mirrors `NDArray_Zeros` for callers that immediately overwrite every
 * element (e.g. arange, the diag/identity diagonal write path). The
 * dtype is validated *before* `Create_NDArray` so a bogus @p type
 * doesn't leak the rtn struct / shape / descriptor through the old
 * `elsize == 0 → return NULL` branch. `rtn->data` is initialised to
 * NULL before any allocator call so that a downstream `NDArray_FREE`
 * is safe even if the device allocation fails halfway.
 *
 * @param[in] shape  Newly-allocated int[ndim]; ownership transfers into
 *                   the returned NDArray's `dimensions`.
 * @param[in] ndim   Number of dimensions (0 yields a 0-D scalar).
 * @param[in] type   Canonical dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray with uninitialised data buffer, or NULL on
 *         validation failure / GPU requested in a non-CUDA build.
 */
NDArray*
NDArray_Empty(int *shape, int ndim, const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) {
            efree(shape);
        }
        return NULL;
    }

    NDArray* rtn = Create_NDArray(shape, ndim, type, NDARRAY_DEVICE_CPU);
    if (rtn == NULL) {
        return NULL;
    }
    /* Ensure NDArray_FREE is safe even if the device allocator below
       returns without writing `rtn->data` (e.g. cudaMalloc failure
       inside `vmalloc`, or an unreachable GPU branch on a non-CUDA build
       reached defensively). */
    rtn->data = NULL;

    size_t bytes = (size_t)NDArray_NUMELEMENTS(rtn) * (size_t)elsize;

    if (device == NDARRAY_DEVICE_CPU) {
        rtn->device = NDARRAY_DEVICE_CPU;
        rtn->data   = emalloc(bytes);
    }
#ifdef HAVE_CUBLAS
    else if (device == NDARRAY_DEVICE_GPU) {
        rtn->device = NDARRAY_DEVICE_GPU;
        vmalloc((void **)&rtn->data, (unsigned int) bytes);
    }
#else
    else if (device == NDARRAY_DEVICE_GPU) {
        /* GPU requested without CUDA — bail loudly rather than hand back
           an NDArray with `data == NULL` and `device == GPU` (which
           every downstream op would dereference). */
        NDArray_FREE(rtn);
        return NULL;
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
 * @brief Allocate a `size × size` identity matrix on the requested device.
 *
 * Always returns a 2-D NDArray of shape `[size, size]`, including the
 * `size == 0` edge case (numpy's `np.identity(0)` is also a `(0, 0)` 2-D
 * array; the previous PHP implementation collapsed this to a 1-D
 * `shape=[0]` array, diverging from numpy).
 *
 * Implementation:
 *  1. Allocate a zero-filled buffer on the target device via
 *     `NDArray_Zeros` — on GPU this is `cudaMalloc` + `cudaMemset(0)`,
 *     so the matrix backing storage exists only in VRAM.
 *  2. Encode the dtype-appropriate `1.0` once into a 16-byte host scratch
 *     (16 = NDARRAY_FP128_SIZE, the widest dtype).
 *  3. Write the diagonal:
 *      - CPU: a `memcpy` loop into the already-zeroed buffer.
 *      - GPU: a single `cudaMemcpy2D` H2D with `dpitch == (size + 1) *
 *        elsize` so each row of the source feeds one diagonal element.
 *        The source is a transient host buffer of `size * elsize` bytes
 *        (≤ 16 × size — trivial for any practical matrix); the result
 *        matrix itself never crosses the bus. For fp128 the host source
 *        carries the (1.0, 0.0) DD pair that the device stores natively.
 *
 * Pre-existing bugs fixed:
 *  - The old implementation hardcoded float32 / CPU storage. Calling it
 *    with any other dtype would have silently written `float` 1.0 into
 *    the (possibly differently-sized) buffer.
 *  - `size == 0` returned a 1-D `[0]` array instead of the canonical
 *    2-D `[0, 0]` matrix.
 *  - The diagonal-index calculation used a redundant
 *    `(i * size * sizeof(float) + i * sizeof(float)) / sizeof(float)`
 *    pattern; now expressed as `i * (size + 1)` element strides
 *    (or `(size + 1) * elsize` bytes for the cudaMemcpy2D pitch).
 *
 * @param[in] size   Side length of the square matrix; must be ≥ 0.
 * @param[in] type   Canonical dtype string (one of the 14 supported aliases).
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New identity NDArray on success, NULL on failure (Error in flight).
 */
NDArray*
NDArray_Identity(int size, const char *type, int device) {
    if (size < 0) {
        zend_throw_error(NULL, "negative dimensions are not allowed");
        return NULL;
    }

    int elsize = get_type_size(type);
    if (elsize == 0) {
        return NULL;
    }

    int *shape = emalloc(sizeof(int) * 2);
    shape[0] = size;
    shape[1] = size;

    NDArray *rtn = NDArray_Zeros(shape, 2, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    if (size == 0) {
        return rtn;
    }

    char encoded[NDARRAY_FP128_SIZE];
    memset(encoded, 0, sizeof(encoded));
    ndarray_set_from_double(type, encoded, 0, 1.0);

    long   n           = (long) size;
    size_t diag_stride = ((size_t)n + 1) * (size_t)elsize;

    if (device == NDARRAY_DEVICE_CPU) {
        char *data = (char *)rtn->data;
        for (long i = 0; i < n; i++) {
            memcpy(data + (size_t)i * diag_stride,
                   encoded, (size_t)elsize);
        }
        return rtn;
    }

#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* fp128 on GPU is DD (hi, lo) — encode (1.0, 0.0) directly so the
           bytes we ship match the on-device layout. Every other dtype
           uses the same byte representation on host and device, so the
           original `encoded` scratch is reused. */
        char one_bytes[NDARRAY_FP128_SIZE];
        if (!strcmp(type, "float128")) {
            double dd[2] = { 1.0, 0.0 };
            memcpy(one_bytes, dd, sizeof(dd));
        } else {
            memcpy(one_bytes, encoded, (size_t)elsize);
        }

        char *src = emalloc((size_t)n * (size_t)elsize);
        if (elsize == 1) {
            memset(src, one_bytes[0], (size_t)n);
        } else {
            for (long i = 0; i < n; i++) {
                memcpy(src + (size_t)i * (size_t)elsize,
                       one_bytes, (size_t)elsize);
            }
        }

        cudaError_t err = cudaMemcpy2D(
            (char *)rtn->data, diag_stride,     /* dst, dpitch */
            src, (size_t)elsize,                /* src, spitch */
            (size_t)elsize, (size_t)n,          /* width, height */
            cudaMemcpyHostToDevice);
        efree(src);
        if (err != cudaSuccess) {
            NDArray_FREE(rtn);
            zend_throw_error(NULL, "cudaMemcpy2D failed: %s",
                             cudaGetErrorString(err));
            return NULL;
        }
        return rtn;
    }
#else
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS. If they didn't,
           fail loudly rather than returning a half-initialised result. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

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
 * @brief Normalise @p a into a fresh NDArray of @p type on @p device.
 *
 * Returns @p a unchanged (and sets `*owned` to 0) when the source already
 * matches the requested dtype/device; otherwise produces a fresh copy
 * (via `NDArray_AsType` for the dtype cast and `NDArray_ToGPU` /
 * `NDArray_ToCPU` for the device move) and sets `*owned` to 1 so the
 * caller knows to release it with `NDArray_FREE`. On failure returns
 * NULL with an Error in flight.
 *
 * @param[in]  a      Source NDArray.
 * @param[in]  type   Canonical target dtype.
 * @param[in]  device Target device id.
 * @param[out] owned  1 if the returned pointer is a fresh copy, 0 otherwise.
 * @return The normalised NDArray, or NULL on failure.
 */
static NDArray *
ndarray_diag_prepare_input(NDArray *a, const char *type, int device, int *owned) {
    *owned = 0;
    NDArray *prep = a;
    if (strcmp(NDArray_TYPE(prep), type) != 0) {
        NDArray *cast = NDArray_AsType(prep, type);
        if (cast == NULL) {
            /* The user's input `a` was never owned by us — leave *owned == 0
               so the caller doesn't try to free it on the NULL return. */
            return NULL;
        }
        prep   = cast;
        *owned = 1;
    }
    if (NDArray_DEVICE(prep) != device) {
#ifdef HAVE_CUBLAS
        NDArray *moved = (device == NDARRAY_DEVICE_GPU)
            ? NDArray_ToGPU(prep) : NDArray_ToCPU(prep);
        if (moved == NULL) {
            /* The cast copy from the previous step (if any) must be
               released before bailing — *owned tracks exactly that. */
            if (*owned) NDArray_FREE(prep);
            *owned = 0;
            return NULL;
        }
        if (*owned) NDArray_FREE(prep);
        prep   = moved;
        *owned = 1;
#else
        if (*owned) NDArray_FREE(prep);
        *owned = 0;
        zend_throw_error(NULL,
            "diag: GPU device requested but CUDA support is not compiled in.");
        return NULL;
#endif
    }
    return prep;
}

/**
 * @brief Build an N×N diagonal matrix whose diagonal is @p prep's 1-D contents.
 *
 * Both @p prep and the result live on @p device with dtype @p type. The
 * zero-fill is delegated to `NDArray_Zeros` (`cudaMalloc` + `cudaMemset(0)`
 * on GPU, `ecalloc` on CPU). The diagonal is then written:
 *  - CPU: per-element `memcpy` from prep[i] to dst[i*(N+1)].
 *  - GPU: a single `cudaMemcpy2D` D2D with `spitch == NDArray_STRIDES(prep)[0]`
 *         and `dpitch == (N+1) * elsize`.
 *
 * @param[in] prep   1-D source NDArray, already in (@p type, @p device).
 * @param[in] type   Canonical dtype string.
 * @param[in] device Target device.
 * @return New N×N NDArray, or NULL on failure.
 */
static NDArray *
ndarray_diag_vector_to_matrix(NDArray *prep, const char *type, int device) {
    long n = NDArray_NUMELEMENTS(prep);
    int *shape = emalloc(sizeof(int) * 2);
    shape[0] = (int)n;
    shape[1] = (int)n;

    NDArray *rtn = NDArray_Zeros(shape, 2, type, device);
    if (rtn == NULL || n == 0) {
        return rtn;
    }

    int    elsize      = NDArray_ELSIZE(rtn);
    size_t diag_stride = ((size_t)n + 1) * (size_t)elsize;
    size_t src_stride  = (size_t) NDArray_STRIDES(prep)[0];

    if (device == NDARRAY_DEVICE_CPU) {
        const char *src = (const char *)NDArray_DATA(prep);
        char       *dst = (char *)NDArray_DATA(rtn);
        for (long i = 0; i < n; i++) {
            memcpy(dst + (size_t)i * diag_stride,
                   src + (size_t)i * src_stride,
                   (size_t)elsize);
        }
        return rtn;
    }
#ifdef HAVE_CUBLAS
    cudaError_t err = cudaMemcpy2D(
        (char *)NDArray_DATA(rtn), diag_stride,
        (const char *)NDArray_DATA(prep), src_stride,
        (size_t)elsize, (size_t)n,
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        NDArray_FREE(rtn);
        zend_throw_error(NULL, "diag: cudaMemcpy2D failed: %s",
                         cudaGetErrorString(err));
        return NULL;
    }
#endif
    return rtn;
}

/**
 * @brief Extract the main diagonal of a 2-D NDArray into a 1-D NDArray.
 *
 * The output length is `min(rows, cols)` — matching numpy. The previous
 * implementation used the input's last dimension directly, which read
 * out of bounds when `rows < cols`.
 *
 * Both @p prep and the result live on @p device with dtype @p type. The
 * diagonal lookup uses `prep`'s own strides so the math is correct even
 * for non-contiguous views: position `(i, i)` sits at byte offset
 * `i * (strides[0] + strides[1])`. On GPU a single `cudaMemcpy2D` D2D
 * does the gather with that stride as `spitch` and `elsize` as `dpitch`.
 *
 * @param[in] prep   2-D source NDArray, already in (@p type, @p device).
 * @param[in] type   Canonical dtype string.
 * @param[in] device Target device.
 * @return New 1-D NDArray of length min(rows, cols), or NULL on failure.
 */
static NDArray *
ndarray_diag_matrix_to_vector(NDArray *prep, const char *type, int device) {
    int rows = NDArray_SHAPE(prep)[0];
    int cols = NDArray_SHAPE(prep)[1];
    int n    = (rows < cols) ? rows : cols;

    int *shape = emalloc(sizeof(int));
    shape[0] = n;

    NDArray *rtn = NDArray_Empty(shape, 1, type, device);
    if (rtn == NULL || n == 0) {
        return rtn;
    }

    int    elsize     = NDArray_ELSIZE(rtn);
    size_t src_stride = (size_t)(NDArray_STRIDES(prep)[0] +
                                 NDArray_STRIDES(prep)[1]);

    if (device == NDARRAY_DEVICE_CPU) {
        const char *src = (const char *)NDArray_DATA(prep);
        char       *dst = (char *)NDArray_DATA(rtn);
        for (long i = 0; i < (long)n; i++) {
            memcpy(dst + (size_t)i * (size_t)elsize,
                   src + (size_t)i * src_stride,
                   (size_t)elsize);
        }
        return rtn;
    }
#ifdef HAVE_CUBLAS
    cudaError_t err = cudaMemcpy2D(
        (char *)NDArray_DATA(rtn), (size_t)elsize,
        (const char *)NDArray_DATA(prep), src_stride,
        (size_t)elsize, (size_t)n,
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        NDArray_FREE(rtn);
        zend_throw_error(NULL, "diag: cudaMemcpy2D failed: %s",
                         cudaGetErrorString(err));
        return NULL;
    }
#endif
    return rtn;
}

/**
 * @brief `NumPower::diag` — dual-mode diagonal constructor / extractor.
 *
 * Mirrors numpy:
 *  - **1-D input** → 2-D `N×N` matrix with @p a on the main diagonal.
 *  - **2-D input** → 1-D vector of `min(rows, cols)` elements holding
 *    the input's main diagonal.
 *
 * The output dtype is @p type and the output device is @p device. If
 * the input doesn't match those it is cast (`NDArray_AsType`) and / or
 * moved (`NDArray_ToGPU` / `NDArray_ToCPU`) into a fresh copy that's
 * freed before this function returns. The actual diagonal traffic is
 * one `cudaMemcpy2D` D2D call on GPU or a tight `memcpy` loop on CPU —
 * see the per-direction helpers for details.
 *
 * Pre-existing bugs fixed:
 *  - Both `NDArray_Diag` and `NDArray_Diagonal` were float32 / CPU
 *    hardcoded; they would read/write 4 bytes per element regardless
 *    of input dtype, corrupting any non-float32 input.
 *  - `NDArray_Diagonal` walked `shape[ndim - 1]` (last dim) as the
 *    diagonal length, which read out of bounds when `rows < cols`
 *    (e.g. a 3×4 matrix would dereference position (3,3)).
 *  - The 2-D-input branch of the old `NDArray_Diag` mutated
 *    `rtn->ndim`, `rtn->dimensions[0]`, and `rtn->strides[0]` after
 *    receiving a 1-D result from `NDArray_Diagonal`, which was already
 *    1-D — the mutations were no-ops at best and corrupted views in
 *    edge cases.
 *
 * @param[in] a      Input NDArray (1-D or 2-D).
 * @param[in] type   Canonical dtype string of the result.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return Newly-allocated result NDArray, or NULL on failure.
 */
NDArray*
NDArray_Diag(NDArray *a, const char *type, int device) {
    int ndim = NDArray_NDIM(a);
    if (ndim != 1 && ndim != 2) {
        zend_throw_error(NULL, "diag: input must be 1-D or 2-D");
        return NULL;
    }
    int elsize = get_type_size(type);
    if (elsize == 0) {
        return NULL;
    }
#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL,
            "diag: GPU device requested but CUDA support is not compiled in.");
        return NULL;
    }
#endif

    int prep_owned;
    NDArray *prep = ndarray_diag_prepare_input(a, type, device, &prep_owned);
    if (prep == NULL) {
        return NULL;
    }

    NDArray *rtn = (ndim == 1)
        ? ndarray_diag_vector_to_matrix(prep, type, device)
        : ndarray_diag_matrix_to_vector(prep, type, device);

    if (prep_owned) {
        NDArray_FREE(prep);
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

/**
 * @brief Return the arange arithmetic kind selected by @p type.
 *
 * Public; see initializers.h for the contract. Used by the PHP layer to
 * decide how to parse the (start, stop, step) zvals before calling
 * `NDArray_Arange`.
 */
NDArrayArangeKind NDArray_ArangeKindFor(const char *type) {
    if (!strcmp(type, "float128")) return NDARRAY_ARANGE_KIND_FP128;
    if (!strcmp(type, "int64"))    return NDARRAY_ARANGE_KIND_INT64;
    if (!strcmp(type, "uint64"))   return NDARRAY_ARANGE_KIND_UINT64;
    return NDARRAY_ARANGE_KIND_DOUBLE;
}

/**
 * @brief Sentinel returned by every `arange_length_*` helper to flag a
 *        step == 0 (or NaN) input or a length that would overflow a long.
 */
#define NDARRAY_ARANGE_LEN_ERROR ((long)-1)

/**
 * @brief Length of the floating-point arange interval @p start..@p stop, step @p step.
 *
 * Every arithmetic kind's length helper returns the non-negative element
 * count, 0 when the sign of @p step is incompatible with `(stop - start)`
 * (mirroring numpy's empty-result behaviour), or
 * `NDARRAY_ARANGE_LEN_ERROR` for degenerate inputs (step == 0, NaN, or
 * a length that would not fit in `long`).
 *
 * @param[in] start First sequence value (inclusive).
 * @param[in] stop  Sequence end (exclusive).
 * @param[in] step  Increment; must be non-zero and finite.
 * @return Element count, 0, or `NDARRAY_ARANGE_LEN_ERROR`.
 */
static long arange_length_double(double start, double stop, double step) {
    if (step == 0.0 || isnan(step) || isnan(start) || isnan(stop)) {
        return NDARRAY_ARANGE_LEN_ERROR;
    }
    double ratio = (stop - start) / step;
    if (!isfinite(ratio) || ratio <= 0.0) {
        return 0;
    }
    double l = ceil(ratio);
    if (l > (double)LONG_MAX) {
        return NDARRAY_ARANGE_LEN_ERROR;
    }
    return (long)l;
}

/**
 * @brief Length of the integer arange interval @p start..@p stop, step @p step.
 *
 * The signed-domain arithmetic that the previous implementation used
 * (`stop - start`, `-step`, `diff + step - 1`) overflows whenever the
 * (start, stop) span exceeds INT64_MAX or @p step is INT64_MIN. The
 * present formulation reinterprets the magnitude in `uint64_t` (well-
 * defined wrap on the int64↔uint64 cast) and ceils via the
 * `(udiff - 1) / abs_step + 1` identity so neither the subtraction nor
 * the ceil can wrap.
 *
 * @param[in] start First sequence value (inclusive).
 * @param[in] stop  Sequence end (exclusive).
 * @param[in] step  Increment; must be non-zero.
 * @return Non-negative element count, 0 when the sign of @p step is
 *         incompatible with `(stop - start)`, or
 *         `NDARRAY_ARANGE_LEN_ERROR` for `step == 0` and any length
 *         that would not fit in `long`.
 */
static long arange_length_int64(int64_t start, int64_t stop, int64_t step) {
    if (step == 0) return NDARRAY_ARANGE_LEN_ERROR;

    uint64_t udiff;
    uint64_t abs_step;
    if (step > 0) {
        if (stop <= start) return 0;
        udiff    = (uint64_t)stop - (uint64_t)start;
        abs_step = (uint64_t)step;
    } else {
        if (stop >= start) return 0;
        udiff    = (uint64_t)start - (uint64_t)stop;
        /* -(uint64_t)step is well-defined even for step == INT64_MIN
           because the unsigned negation wraps to 2^63 — exactly the
           magnitude of INT64_MIN. */
        abs_step = -(uint64_t)step;
    }

    /* `(udiff - 1) / abs_step + 1` is the ceil-without-overflow identity
       (udiff > 0 here, so udiff - 1 doesn't wrap). The classic
       `(udiff + abs_step - 1) / abs_step` would overflow when udiff is
       close to UINT64_MAX. */
    uint64_t length = (udiff - 1) / abs_step + 1;
    if (length > (uint64_t)LONG_MAX) {
        return NDARRAY_ARANGE_LEN_ERROR;
    }
    return (long)length;
}

/**
 * @brief Length of the unsigned arange interval @p start..@p stop, step @p step.
 *
 * Companion to `arange_length_int64` — same overflow-safe ceiling-
 * division identity (`(udiff - 1) / step + 1`) so that the result
 * doesn't wrap when @p stop is close to `UINT64_MAX`.
 *
 * @param[in] start First sequence value (inclusive).
 * @param[in] stop  Sequence end (exclusive).
 * @param[in] step  Increment; must be non-zero and is unsigned, so the
 *                  sequence is monotonically increasing.
 * @return Non-negative element count, 0 when `stop <= start`, or
 *         `NDARRAY_ARANGE_LEN_ERROR` for `step == 0` / overflow.
 */
static long arange_length_uint64(uint64_t start, uint64_t stop, uint64_t step) {
    if (step == 0) return NDARRAY_ARANGE_LEN_ERROR;
    if (stop <= start) return 0;
    uint64_t udiff  = stop - start;
    uint64_t length = (udiff - 1) / step + 1;
    if (length > (uint64_t)LONG_MAX) {
        return NDARRAY_ARANGE_LEN_ERROR;
    }
    return (long)length;
}

/**
 * @brief Length of the fp128 arange interval @p start..@p stop, step @p step.
 *
 * Uses the platform-uniform `NDARRAY_FP128_*` macros so the same source
 * compiles for both the native `__float128` backend and the
 * double-double fallback. The element count is computed as
 * `ceil((stop - start) / step)`; rounding goes through a `double` cast
 * because every legal count is bounded by `LONG_MAX < 2^63` which is
 * exactly representable in a double.
 *
 * @param[in] start First sequence value (inclusive).
 * @param[in] stop  Sequence end (exclusive).
 * @param[in] step  Increment; must be non-zero.
 * @return Element count, 0, or `NDARRAY_ARANGE_LEN_ERROR`.
 */
static long arange_length_fp128(ndarray_fp128_t start,
                                ndarray_fp128_t stop,
                                ndarray_fp128_t step) {
    if (NDARRAY_FP128_ISZERO(step)) return NDARRAY_ARANGE_LEN_ERROR;
    ndarray_fp128_t ratio = NDARRAY_FP128_DIV(
        NDARRAY_FP128_SUB(stop, start), step);
    /* ratio < 0 (sign mismatch) → empty result, matching numpy. */
    if (NDARRAY_FP128_LT(ratio, NDARRAY_FP128_ZERO()) ||
        NDARRAY_FP128_ISZERO(ratio)) {
        return 0;
    }
    /* The ceil is taken on double — `ratio` already represents an integer
       count (bounded by LONG_MAX << 2^53), so the round-trip through
       double is exact. */
    double rd = NDARRAY_FP128_TO_D(ratio);
    if (!isfinite(rd) || rd > (double)LONG_MAX) {
        return NDARRAY_ARANGE_LEN_ERROR;
    }
    return (long)ceil(rd);
}

/**
 * @brief Write @p n arange elements into a host buffer with double precision.
 *
 * Used for every dtype except `float128`, `int64`, and `uint64`: a
 * `double` covers each of those types' representable range exactly, and
 * the per-element cast / quantisation is delegated to
 * `ndarray_set_from_double`. Each value is computed by the closed-form
 * `start + i * step` so the result doesn't accumulate rounding error
 * across the array.
 *
 * @param[out] out   Host buffer of `n * get_type_size(type)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  type  Canonical dtype string (used by `ndarray_set_from_double`).
 * @param[in]  start First sequence value.
 * @param[in]  step  Increment.
 */
static void arange_fill_double(char *out, long n, const char *type,
                                double start, double step) {
    for (long i = 0; i < n; i++) {
        ndarray_set_from_double(type, out, (size_t)i,
                                start + (double)i * step);
    }
}

/**
 * @brief Write @p n arange elements into a host buffer with fp128 precision.
 *
 * Uses the platform-uniform `NDARRAY_FP128_*` macros so the same code
 * runs on both the native `__float128` backend and the double-double
 * fallback. Each element is `start + i * step` in fp128 — no cumulative
 * add so error doesn't accumulate.
 *
 * @param[out] out   Host buffer of `n * NDARRAY_FP128_SIZE` bytes.
 * @param[in]  n     Element count.
 * @param[in]  start First sequence value in host fp128 representation.
 * @param[in]  step  Increment.
 */
static void arange_fill_fp128(char *out, long n,
                               ndarray_fp128_t start,
                               ndarray_fp128_t step) {
    ndarray_fp128_t *p = (ndarray_fp128_t *)out;
    for (long i = 0; i < n; i++) {
        ndarray_fp128_t inc = NDARRAY_FP128_MUL(
            NDARRAY_FP128_FROM_I64((int64_t)i), step);
        p[i] = NDARRAY_FP128_ADD(start, inc);
    }
}

/**
 * @brief Write @p n arange elements into a host buffer as `int64_t`.
 *
 * `start + i * step` cannot overflow under any legal input: the length
 * helper guarantees the last value (`start + (n - 1) * step`) lies
 * strictly between `start` and `stop`, so the magnitude stays inside
 * the int64 range.
 *
 * @param[out] out   Host buffer of `n * sizeof(int64_t)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  start First sequence value.
 * @param[in]  step  Increment.
 */
static void arange_fill_int64(char *out, long n, int64_t start, int64_t step) {
    int64_t *p = (int64_t *)out;
    for (long i = 0; i < n; i++) {
        p[i] = start + (int64_t)i * step;
    }
}

/**
 * @brief Write @p n arange elements into a host buffer as `uint64_t`.
 *
 * Companion to `arange_fill_int64`. Per-element value is
 * `start + i * step`; the unsigned arithmetic is well-defined and the
 * length helper guarantees `start + (n - 1) * step < stop`, so the
 * result never wraps.
 *
 * @param[out] out   Host buffer of `n * sizeof(uint64_t)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  start First sequence value.
 * @param[in]  step  Increment.
 */
static void arange_fill_uint64(char *out, long n, uint64_t start, uint64_t step) {
    uint64_t *p = (uint64_t *)out;
    for (long i = 0; i < n; i++) {
        p[i] = start + (uint64_t)i * step;
    }
}

/**
 * @brief Build a 1-D arange NDArray of the requested dtype on the requested device.
 *
 * Each dtype follows the arithmetic kind dictated by `NDArray_ArangeKindFor`:
 *  - `float128`   → fp128 arithmetic (113-bit / DD).
 *  - `int64`      → 64-bit signed integer arithmetic.
 *  - `uint64`     → 64-bit unsigned integer arithmetic.
 *  - everything else → double arithmetic.
 *
 * The host fill is computed by closed form `a[i] = start + i * step` (not
 * cumulative add) so floating-point dtypes don't drift across the array.
 * For GPU the closed-form result is computed on host and then handed to
 * `NDArray_TypedH2D`, which converts host __float128/DD bytes into the
 * on-device DD pair for fp128. The host scratch is freed on every exit.
 *
 * Pre-existing bugs fixed:
 *  - The old `NDArray_Arange(double, double, double)` was float32 / CPU
 *    hardcoded and would have allocated `numElements * sizeof(float)`
 *    bytes regardless of dtype.
 *  - It threw on `length == 0` instead of returning the empty array
 *    numpy produces for `arange(5, 0, 1)`.
 *  - The element generation used cumulative add (`a[i] = a[i-1] + step`),
 *    which accumulates rounding error proportional to the array length.
 *  - The wasted `NDArray_Zeros` (full memset) before overwriting every
 *    element is gone — `NDArray_Empty` allocates uninitialised storage.
 *
 * @param[in] spec   Discriminated (start, stop, step) triple in the
 *                   arithmetic kind dictated by @p type.
 * @param[in] type   Canonical NDArray dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New 1-D arange NDArray, or NULL on failure (Error in flight).
 */
NDArray*
NDArray_Arange(const NDArrayArangeSpec *spec, const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        return NULL;
    }

    long length = NDARRAY_ARANGE_LEN_ERROR;
    switch (spec->kind) {
        case NDARRAY_ARANGE_KIND_DOUBLE:
            length = arange_length_double(
                spec->v.d.start, spec->v.d.stop, spec->v.d.step);
            break;
        case NDARRAY_ARANGE_KIND_FP128:
            length = arange_length_fp128(
                spec->v.f128.start, spec->v.f128.stop, spec->v.f128.step);
            break;
        case NDARRAY_ARANGE_KIND_INT64:
            length = arange_length_int64(
                spec->v.i64.start, spec->v.i64.stop, spec->v.i64.step);
            break;
        case NDARRAY_ARANGE_KIND_UINT64:
            length = arange_length_uint64(
                spec->v.u64.start, spec->v.u64.stop, spec->v.u64.step);
            break;
    }

    if (length == NDARRAY_ARANGE_LEN_ERROR) {
        zend_throw_error(NULL,
            "arange: step must be non-zero and finite");
        return NULL;
    }
    if (length > (long)INT_MAX) {
        zend_throw_error(NULL,
            "arange: computed length %ld exceeds INT_MAX", length);
        return NULL;
    }

    int *shape = emalloc(sizeof(int));
    shape[0] = (int)length;

    NDArray *rtn = NDArray_Empty(shape, 1, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    if (length == 0) {
        return rtn;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    /* CPU writes the closed-form values straight into NDArray_DATA. GPU
       writes to a host staging buffer first (size n * elsize, freed on
       exit), then `NDArray_TypedH2D` ships the bytes to VRAM — handling
       the host-fp128 → on-device DD conversion for the fp128 case. The
       result matrix itself is built directly in VRAM. */
    char *host_data = (char *)rtn->data;
    int   to_gpu    = 0;
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        host_data = emalloc((size_t)length * (size_t)elsize);
        to_gpu    = 1;
    }
#endif

    switch (spec->kind) {
        case NDARRAY_ARANGE_KIND_DOUBLE:
            arange_fill_double(host_data, length, type,
                               spec->v.d.start, spec->v.d.step);
            break;
        case NDARRAY_ARANGE_KIND_FP128:
            arange_fill_fp128(host_data, length,
                              spec->v.f128.start, spec->v.f128.step);
            break;
        case NDARRAY_ARANGE_KIND_INT64:
            arange_fill_int64(host_data, length,
                              spec->v.i64.start, spec->v.i64.step);
            break;
        case NDARRAY_ARANGE_KIND_UINT64:
            arange_fill_uint64(host_data, length,
                               spec->v.u64.start, spec->v.u64.step);
            break;
    }

#ifdef HAVE_CUBLAS
    if (to_gpu) {
        NDArray_TypedH2D((char *)rtn->data, host_data, length, type);
        efree(host_data);
    }
#endif

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
