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

/* The truncated-normal sampler shares the (loc, scale) coercion plumbing
   and the per-dtype dispatch with `NDArray_Normal` — every helper below
   reuses the same `NDArrayNormalSpec` discriminator. The full
   per-element fillers are defined further down, next to the unbounded
   `NDArray_Normal` path; the truncated variant just adds a rejection
   guard on top. */

/**
 * @brief Sample one standard-normal `double` z ~ N(0, 1).
 *
 * Marsaglia / polar variant of the Box-Muller transform: two uniform
 * draws map to two independent standard normals. Each call returns one
 * sample; the helper caches the second of every pair so the cost
 * amortises to ~one polar iteration per call. State is `static`
 * (thread-local would require a runtime guarantee we don't make), which
 * mirrors the project's existing CPU rng usage in `NDArray_Poisson`,
 * `NDArray_TruncatedNormal`, etc.
 *
 * @return One sample drawn from N(0, 1).
 */
static double ndarray_normal_sample(void) {
    static int    have_spare = 0;
    static double spare;
    if (have_spare) {
        have_spare = 0;
        return spare;
    }
    double u, v, s;
    do {
        u = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
        v = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double factor = sqrt(-2.0 * log(s) / s);
    spare      = v * factor;
    have_spare = 1;
    return u * factor;
}

/**
 * @brief Map a canonical dtype string to its normal-sampler arithmetic kind.
 *
 * `float128` returns FP128 so loc/scale stay at native fp128 precision;
 * `uint64` returns UINT64 so means past 2^53 stay exact; every other
 * dtype routes through `double`, which already covers each smaller
 * dtype's representable range.
 *
 * @param[in] type Canonical dtype string.
 * @return one of NDARRAY_NORMAL_KIND_*.
 */
NDArrayNormalKind NDArray_NormalKindFor(const char *type) {
    if (!strcmp(type, "float128")) return NDARRAY_NORMAL_KIND_FP128;
    if (!strcmp(type, "uint64"))   return NDARRAY_NORMAL_KIND_UINT64;
    return NDARRAY_NORMAL_KIND_DOUBLE;
}

/**
 * @brief CPU fill: write @p n N(loc, scale^2) samples into @p data of @p type.
 *
 * Uses double-precision Box-Muller (`ndarray_normal_sample`) and routes
 * the per-element store through `ndarray_set_from_double` so each dtype
 * gets its dtype-correct quantisation (fp4 LUT pick, integer truncation,
 * etc.). Covers every dtype except fp128 (DD precision) and uint64
 * (full unsigned-64-bit range) — both have dedicated fillers below.
 *
 * @param[out] data  Destination host buffer; ≥ `n * get_type_size(type)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  type  Canonical dtype string.
 * @param[in]  loc   Distribution mean (µ).
 * @param[in]  scale Distribution stddev (σ).
 */
static void normal_fill_cpu_double(char *data, long n, const char *type,
                                    double loc, double scale) {
    for (long i = 0; i < n; i++) {
        double v = loc + scale * ndarray_normal_sample();
        ndarray_set_from_double(type, data, (size_t)i, v);
    }
}

/**
 * @brief CPU fill for the float128 dtype.
 *
 * Computes each value as `value = loc + scale * fp128(z)` in fp128
 * arithmetic so wide-range loc/scale (`'1e200'` via string input) keep
 * their full precision. The z-sample itself is a double (53 bits) —
 * matches CPU and GPU paths, where the underlying PRNG is fp64.
 *
 * @param[out] data  Destination host buffer of `n * NDARRAY_FP128_SIZE` bytes.
 * @param[in]  n     Element count.
 * @param[in]  loc   Distribution mean as fp128.
 * @param[in]  scale Distribution stddev as fp128.
 */
static void normal_fill_cpu_fp128(char *data, long n,
                                   ndarray_fp128_t loc,
                                   ndarray_fp128_t scale) {
    ndarray_fp128_t *p = (ndarray_fp128_t *)data;
    for (long i = 0; i < n; i++) {
        ndarray_fp128_t z = NDARRAY_FP128_FROM_D(ndarray_normal_sample());
        p[i] = NDARRAY_FP128_ADD(loc, NDARRAY_FP128_MUL(scale, z));
    }
}

/**
 * @brief CPU fill for the uint64 dtype.
 *
 * Computes `value = loc + (int64_t)(scale_d * z)` where `scale_d` is the
 * double cast of `scale` (uint64). The signed cast lets negative-z
 * shifts subtract from `loc`; the unsigned modular arithmetic in C
 * preserves the lossless `2^64` wraparound, matching numpy's
 * `astype(uint64)` semantics for negative floats (well-defined as a
 * conversion, not as a clip).
 *
 * Loc/scale themselves are kept in `uint64_t` so a caller's
 * `'18446744073709551615'` mean survives PHP's IS_STRING path; the
 * actual stochastic noise is dominated by `scale`, which a user
 * typically picks smaller than 2^53 anyway (RNG entropy ceiling).
 *
 * @param[out] data  Destination host buffer of `n * sizeof(uint64_t)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  loc   Distribution mean as uint64.
 * @param[in]  scale Distribution stddev as uint64.
 */
static void normal_fill_cpu_uint64(char *data, long n,
                                    uint64_t loc, uint64_t scale) {
    uint64_t *p = (uint64_t *)data;
    double   scale_d = (double)scale;
    for (long i = 0; i < n; i++) {
        double  z       = ndarray_normal_sample();
        int64_t delta_s = (int64_t)(scale_d * z);
        p[i] = loc + (uint64_t)delta_s;
    }
}

/**
 * @brief Split an fp128 value into its (hi, lo) double-double pair.
 *
 * Mirrors the conversion in `ndarray_broadcast_encoded` (the fill path
 * for `Ones` / `Full`). On the native-fp128 backend the lo word is the
 * residue `value - (double)hi`; on the DD fallback the struct already
 * stores the pair. ±INF and NaN forward as `(hi, 0.0)`.
 *
 * @param[in]  v   fp128 source value.
 * @param[out] hi  high double.
 * @param[out] lo  low double.
 */
static void ndarray_fp128_split(ndarray_fp128_t v, double *hi, double *lo) {
#if NDARRAY_HAVE_FLOAT128
    double h = (double)v;
    *hi = h;
    *lo = (isinf(h) || isnan(h)) ? 0.0 : (double)(v - (ndarray_fp128_t)h);
#else
    *hi = v.hi;
    *lo = (isinf(v.hi) || isnan(v.hi)) ? 0.0 : v.lo;
#endif
}

/**
 * @brief Build a Gaussian-sample NDArray of the requested shape / dtype / device.
 *
 * Dispatches by the discriminated @p spec — `DOUBLE`, `FP128`, or
 * `UINT64` — and by @p device. For `device == NDARRAY_DEVICE_GPU` the
 * destination buffer is allocated directly in VRAM via `NDArray_Empty`
 * (no host-side staging of the result):
 *
 *  - `dtype == float32` / `float64`: cuRAND fills the destination
 *    in-place via `cuda_normal_f32` / `cuda_normal_f64`.
 *  - Other dtypes with `DOUBLE` arithmetic: cuRAND fills a transient
 *    GPU `float32` scratch, then a single `cuda_cast_f32_to_<dst>`
 *    quantises into the destination. Scratch is freed before return.
 *  - `int64` / dtypes with wider mantissas: cuRAND fills a transient
 *    GPU `float64` scratch, then `cuda_cast_f64_to_<dst>` quantises.
 *  - `dtype == float128`: cuRAND fills a GPU `float64` z-scratch, then
 *    a custom DD-affine kernel (`cuda_normal_dd_affine`) computes
 *    `loc + scale * z` in true double-double arithmetic on device,
 *    storing the (hi, lo) pair into the destination DD buffer.
 *  - `dtype == uint64` with `UINT64` kind: cuRAND fills a `float64`
 *    z-scratch via `cuda_normal_f64`, then `cuda_normal_u64_affine`
 *    writes `loc + (uint64_t)((int64_t)(scale * z))` directly into the
 *    destination — entire pipeline stays VRAM-side, no host staging of
 *    the result.
 *
 * For `device == NDARRAY_DEVICE_CPU` every kind writes straight into
 * the destination host buffer via the per-kind fillers above.
 *
 * Pre-existing bugs fixed:
 *  - The old GPU path leaked the just-allocated `NDArray_Zeros` buffer
 *    (allocated via `vmalloc`, then immediately `vfree`d before being
 *    replaced with a fresh `d_matrix` — two allocations, one freed
 *    inside a path that also goes through `cudaMemset(0)` on the
 *    discarded buffer for no reason).
 *  - The seed was pinned to `1234ULL`, so every call in the same process
 *    produced identical samples — fixed by `cuda_normal_next_seed`.
 *  - The CPU loop's outer `for (i; i < size; i += 2)` write to
 *    `[i + 1]` was guarded by `i + 1 < size`, but the inner polar
 *    rejection ignored the case where `size == 0` could still enter
 *    the body; the per-sample helper avoids that whole shape.
 *  - Dtype was hardcoded float32 / CPU regardless of caller request.
 *
 * @param[in] spec   Discriminated (loc, scale) pair in the kind dictated
 *                   by @p type.
 * @param[in] shape  Newly-allocated `int[ndim]`; ownership transfers
 *                   into the returned NDArray's `dimensions`.
 * @param[in] ndim   Number of dimensions; 0 yields a 0-D scalar.
 * @param[in] type   Canonical NDArray dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray on success, NULL on failure (Error in flight).
 */
NDArray*
NDArray_Normal(const NDArrayNormalSpec *spec, int *shape, int ndim,
               const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) efree(shape);
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    long n = (long) NDArray_NUMELEMENTS(rtn);
    if (n <= 0) {
        return rtn;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before requesting
           GPU. If they didn't, fail loudly instead of returning an
           NDArray with uninitialised on-device storage. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    if (device == NDARRAY_DEVICE_CPU) {
        switch (spec->kind) {
            case NDARRAY_NORMAL_KIND_DOUBLE:
                normal_fill_cpu_double((char *)rtn->data, n, type,
                                        spec->v.d.loc, spec->v.d.scale);
                break;
            case NDARRAY_NORMAL_KIND_FP128:
                normal_fill_cpu_fp128((char *)rtn->data, n,
                                       spec->v.f128.loc, spec->v.f128.scale);
                break;
            case NDARRAY_NORMAL_KIND_UINT64:
                normal_fill_cpu_uint64((char *)rtn->data, n,
                                        spec->v.u64.loc, spec->v.u64.scale);
                break;
        }
        return rtn;
    }

#ifdef HAVE_CUBLAS
    /* From here on: device == NDARRAY_DEVICE_GPU. The destination buffer
       lives in VRAM (NDArray_Empty allocated it via vmalloc/cudaMalloc).
       Each branch writes into it without ever copying the result through
       host memory. */
    if (spec->kind == NDARRAY_NORMAL_KIND_DOUBLE) {
        double loc   = spec->v.d.loc;
        double scale = spec->v.d.scale;

        if (!strcmp(type, "float32")) {
            cuda_normal_f32((float *)rtn->data, n, (float)loc, (float)scale);
            return rtn;
        }
        if (!strcmp(type, "float64")) {
            cuda_normal_f64((double *)rtn->data, n, loc, scale);
            return rtn;
        }

        /* Every other dtype in the DOUBLE kind gets a transient GPU f32
           scratch filled by cuRAND, then a single cuda_cast_f32_to_<dst>
           pass quantises into the destination. The scratch lives only
           inside this call. */
        float *scratch = NULL;
        vmalloc((void **)&scratch, (unsigned int)((size_t)n * sizeof(float)));
        if (scratch == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_normal_f32(scratch, n, (float)loc, (float)scale);

        if (!strcmp(type, "float4")) {
            cuda_cast_f32_to_fp4(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "float8")) {
            cuda_cast_f32_to_fp8(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "float16")) {
            cuda_cast_f32_to_f16(scratch, (uint16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int8")) {
            cuda_cast_f32_to_i8(scratch, (int8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint8")) {
            cuda_cast_f32_to_u8(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int16")) {
            cuda_cast_f32_to_i16(scratch, (int16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint16")) {
            cuda_cast_f32_to_u16(scratch, (uint16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int32")) {
            cuda_cast_f32_to_i32(scratch, (int32_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint32")) {
            cuda_cast_f32_to_u32(scratch, (uint32_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int64")) {
            /* int64 has range past float32's 24-bit mantissa. Promote
               the scratch to fp64 before casting so values near 2^31..2^63
               survive without aliasing. */
            double *scratch64 = NULL;
            vmalloc((void **)&scratch64,
                    (unsigned int)((size_t)n * sizeof(double)));
            if (scratch64 != NULL) {
                cuda_normal_f64(scratch64, n, loc, scale);
                cuda_cast_f64_to_i64(scratch64, (int64_t *)rtn->data, (int)n);
                vfree(scratch64);
            }
        } else {
            /* Should be unreachable: NDArray_NormalKindFor only returns
               DOUBLE for these dtypes. */
            vfree(scratch);
            NDArray_FREE(rtn);
            return NULL;
        }
        vfree(scratch);
        return rtn;
    }

    if (spec->kind == NDARRAY_NORMAL_KIND_FP128) {
        /* Generate fp64 z-samples on GPU, then a DD-affine kernel writes
           the result DD pair into the destination — entire pipeline
           stays VRAM-side; loc/scale go to the device as scalar args. */
        double *z = NULL;
        vmalloc((void **)&z, (unsigned int)((size_t)n * sizeof(double)));
        if (z == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_normal_f64(z, n, 0.0, 1.0);

        double loc_hi, loc_lo, scale_hi, scale_lo;
        ndarray_fp128_split(spec->v.f128.loc, &loc_hi, &loc_lo);
        ndarray_fp128_split(spec->v.f128.scale, &scale_hi, &scale_lo);

        cuda_normal_dd_affine(z, (double *)rtn->data, n,
                               loc_hi, loc_lo, scale_hi, scale_lo);
        vfree(z);
        return rtn;
    }

    if (spec->kind == NDARRAY_NORMAL_KIND_UINT64) {
        /* Generate fp64 standard-normal samples on GPU, then a u64 affine
           kernel writes `loc + (uint64_t)((int64_t)(scale * z))` directly
           into the destination — entire pipeline stays VRAM-side. */
        double *z = NULL;
        vmalloc((void **)&z, (unsigned int)((size_t)n * sizeof(double)));
        if (z == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_normal_f64(z, n, 0.0, 1.0);

        cuda_normal_u64_affine(z, (unsigned long long *)rtn->data, n,
                                (unsigned long long)spec->v.u64.loc,
                                (double)spec->v.u64.scale);
        vfree(z);
        return rtn;
    }
#endif

    /* Defensive: every reachable (device, kind) combination is handled
       above. If we get here something is mis-wired — free and bail. */
    NDArray_FREE(rtn);
    return NULL;
}

/**
 * @brief Sample one truncated standard-normal `double` z ~ N(0, 1) | |z| ≤ 2.
 *
 * Rejection-sampling wrapper around `ndarray_normal_sample`: keeps
 * drawing until the sample lands in `[-2, 2]`. The acceptance rate at
 * ±2σ is ~95.45%, so the average runtime is ~1.05 calls to the
 * underlying polar sampler per accepted value — well bounded.
 *
 * @return One sample drawn from N(0, 1) truncated to [-2, 2].
 */
static double ndarray_truncated_normal_sample(void) {
    double z;
    do {
        z = ndarray_normal_sample();
    } while (z < -2.0 || z > 2.0);
    return z;
}

/**
 * @brief CPU fill: write @p n truncated-Gaussian samples into @p data.
 *
 * Companion to `normal_fill_cpu_double`; samples are rejection-bounded
 * to `[loc - 2σ, loc + 2σ]` (the truncation is on the standardised z,
 * which is then affinely transformed). Routes the per-element store
 * through `ndarray_set_from_double` so each dtype gets its
 * dtype-correct quantisation.
 *
 * @param[out] data  Destination host buffer; ≥ `n * get_type_size(type)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  type  Canonical dtype string.
 * @param[in]  loc   Distribution mean (µ).
 * @param[in]  scale Distribution stddev (σ).
 */
static void truncated_normal_fill_cpu_double(char *data, long n,
                                              const char *type,
                                              double loc, double scale) {
    for (long i = 0; i < n; i++) {
        double v = loc + scale * ndarray_truncated_normal_sample();
        ndarray_set_from_double(type, data, (size_t)i, v);
    }
}

/**
 * @brief CPU fill for the float128 dtype.
 *
 * Same precision discipline as `normal_fill_cpu_fp128` — the affine
 * `loc + scale * fp128(z)` runs in fp128 arithmetic so wide-range
 * loc/scale (`'1e+200'` via string input) keep full precision. The
 * z-sample is a (truncated) standard-normal double.
 *
 * @param[out] data  Destination host buffer of `n * NDARRAY_FP128_SIZE` bytes.
 * @param[in]  n     Element count.
 * @param[in]  loc   Distribution mean as fp128.
 * @param[in]  scale Distribution stddev as fp128.
 */
static void truncated_normal_fill_cpu_fp128(char *data, long n,
                                             ndarray_fp128_t loc,
                                             ndarray_fp128_t scale) {
    ndarray_fp128_t *p = (ndarray_fp128_t *)data;
    for (long i = 0; i < n; i++) {
        ndarray_fp128_t z = NDARRAY_FP128_FROM_D(ndarray_truncated_normal_sample());
        p[i] = NDARRAY_FP128_ADD(loc, NDARRAY_FP128_MUL(scale, z));
    }
}

/**
 * @brief CPU fill for the uint64 dtype.
 *
 * Companion to `normal_fill_cpu_uint64`: the standardised z is bounded
 * to `[-2, 2]`, so the signed delta `(int64_t)(scale_d * z)` lies in
 * `[-2·scale, 2·scale]`. The `uint64` modular addition preserves the
 * 2^64 wrap if the user picks a tiny `loc` and a `scale` larger than
 * `loc / 2`; callers near the boundary should size their parameters
 * accordingly.
 *
 * @param[out] data  Destination host buffer of `n * sizeof(uint64_t)` bytes.
 * @param[in]  n     Element count.
 * @param[in]  loc   Distribution mean as uint64.
 * @param[in]  scale Distribution stddev as uint64.
 */
static void truncated_normal_fill_cpu_uint64(char *data, long n,
                                              uint64_t loc, uint64_t scale) {
    uint64_t *p = (uint64_t *)data;
    double   scale_d = (double)scale;
    for (long i = 0; i < n; i++) {
        double  z       = ndarray_truncated_normal_sample();
        int64_t delta_s = (int64_t)(scale_d * z);
        p[i] = loc + (uint64_t)delta_s;
    }
}

/**
 * @brief Build a truncated-Gaussian NDArray of the requested shape / dtype / device.
 *
 * Mirrors `NDArray_Normal` exactly — same `NDArrayNormalSpec`
 * discriminator, same CPU fill pattern, same GPU VRAM-direct
 * pipeline — except every per-element draw is rejection-sampled so the
 * accepted `(z - 0) / 1` value lies in `[-2, 2]` and the stored value
 * therefore lies in `[loc - 2σ, loc + 2σ]`.
 *
 * GPU dispatch:
 *  - `dtype == float32`: `cuda_truncated_normal_f32` fills directly.
 *  - `dtype == float64`: `cuda_truncated_normal_f64` fills directly.
 *  - Other DOUBLE-kind dtypes: transient f32 scratch filled by
 *    `cuda_truncated_normal_f32`, then `cuda_cast_f32_to_<dst>` quantises.
 *    `int64` uses an f64 scratch for the same reason as `NDArray_Normal`
 *    (avoids float32 mantissa aliasing past 2^24).
 *  - `dtype == float128`: f64 scratch holds standard-truncated z-samples
 *    (loc = 0, scale = 1), then `cuda_normal_dd_affine` applies the
 *    user's fp128 loc/scale in DD arithmetic on device.
 *  - `dtype == uint64`: `cuda_truncated_normal_f64` fills a z-scratch
 *    of standard truncated-normal samples, then the shared
 *    `cuda_normal_u64_affine` kernel writes
 *    `loc + (uint64_t)((int64_t)(scale * z))` directly into the
 *    destination — entire pipeline stays VRAM-side.
 *
 * Pre-existing bugs fixed:
 *  - Silent `scale = scale / 0.88` rescale that mutated the user's σ
 *    before sampling and again before the truncation check — removed,
 *    so `scale` now means what the caller passed.
 *  - Pinned `1234ULL` cuRAND seed → identical samples on every call;
 *    now `cuda_normal_next_seed` gives a fresh per-call stream.
 *  - Dtype was hardcoded float32 / CPU regardless of the caller's
 *    request, ignoring loc/scale precision for fp128 / uint64.
 *  - CPU loop wrote to `rtn->data[i]` *inside* the do-while reject
 *    body, doing one or more wasted writes per accepted sample. The
 *    per-sample helper resolves this naturally.
 *  - GPU kernel used a cos-only Box-Muller variant via `curand_normal`
 *    — fine, but combined with the pinned seed it was deterministic.
 *
 * @param[in] spec   Discriminated (loc, scale) pair in the kind
 *                   dictated by @p type.
 * @param[in] shape  Newly-allocated `int[ndim]`; ownership transfers
 *                   into the returned NDArray's `dimensions`.
 * @param[in] ndim   Number of dimensions; 0 yields a 0-D scalar.
 * @param[in] type   Canonical NDArray dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray on success, NULL on failure (Error in flight).
 */
NDArray*
NDArray_TruncatedNormal(const NDArrayNormalSpec *spec, int *shape, int ndim,
                        const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) efree(shape);
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    long n = (long) NDArray_NUMELEMENTS(rtn);
    if (n <= 0) {
        return rtn;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before requesting
           GPU. If they didn't, fail loudly instead of returning an
           NDArray with uninitialised on-device storage. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    if (device == NDARRAY_DEVICE_CPU) {
        switch (spec->kind) {
            case NDARRAY_NORMAL_KIND_DOUBLE:
                truncated_normal_fill_cpu_double((char *)rtn->data, n, type,
                                                  spec->v.d.loc,
                                                  spec->v.d.scale);
                break;
            case NDARRAY_NORMAL_KIND_FP128:
                truncated_normal_fill_cpu_fp128((char *)rtn->data, n,
                                                 spec->v.f128.loc,
                                                 spec->v.f128.scale);
                break;
            case NDARRAY_NORMAL_KIND_UINT64:
                truncated_normal_fill_cpu_uint64((char *)rtn->data, n,
                                                  spec->v.u64.loc,
                                                  spec->v.u64.scale);
                break;
        }
        return rtn;
    }

#ifdef HAVE_CUBLAS
    /* device == NDARRAY_DEVICE_GPU. The destination buffer lives in VRAM
       (NDArray_Empty allocated it via vmalloc/cudaMalloc); every branch
       below writes into it without copying the result through host
       memory. */
    if (spec->kind == NDARRAY_NORMAL_KIND_DOUBLE) {
        double loc   = spec->v.d.loc;
        double scale = spec->v.d.scale;

        if (!strcmp(type, "float32")) {
            cuda_truncated_normal_f32((float *)rtn->data, n,
                                       (float)loc, (float)scale);
            return rtn;
        }
        if (!strcmp(type, "float64")) {
            cuda_truncated_normal_f64((double *)rtn->data, n, loc, scale);
            return rtn;
        }

        /* Other dtypes in the DOUBLE kind: generate truncated samples
           into a transient GPU f32 scratch, then quantise via
           `cuda_cast_f32_to_<dst>`. */
        float *scratch = NULL;
        vmalloc((void **)&scratch, (unsigned int)((size_t)n * sizeof(float)));
        if (scratch == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_truncated_normal_f32(scratch, n, (float)loc, (float)scale);

        if (!strcmp(type, "float4")) {
            cuda_cast_f32_to_fp4(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "float8")) {
            cuda_cast_f32_to_fp8(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "float16")) {
            cuda_cast_f32_to_f16(scratch, (uint16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int8")) {
            cuda_cast_f32_to_i8(scratch, (int8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint8")) {
            cuda_cast_f32_to_u8(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int16")) {
            cuda_cast_f32_to_i16(scratch, (int16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint16")) {
            cuda_cast_f32_to_u16(scratch, (uint16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int32")) {
            cuda_cast_f32_to_i32(scratch, (int32_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint32")) {
            cuda_cast_f32_to_u32(scratch, (uint32_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int64")) {
            /* int64 with values past 2^24 would alias through float32;
               regenerate at fp64 precision so the truncation window is
               preserved (the [-2σ, +2σ] check is exact in double for
               any |loc|, |scale| < 2^53). */
            double *scratch64 = NULL;
            vmalloc((void **)&scratch64,
                    (unsigned int)((size_t)n * sizeof(double)));
            if (scratch64 != NULL) {
                cuda_truncated_normal_f64(scratch64, n, loc, scale);
                cuda_cast_f64_to_i64(scratch64, (int64_t *)rtn->data, (int)n);
                vfree(scratch64);
            }
        } else {
            /* Unreachable: NDArray_NormalKindFor only returns DOUBLE for
               these dtypes; any other dtype routes to FP128 / UINT64. */
            vfree(scratch);
            NDArray_FREE(rtn);
            return NULL;
        }
        vfree(scratch);
        return rtn;
    }

    if (spec->kind == NDARRAY_NORMAL_KIND_FP128) {
        /* Generate standardised truncated-normal z-samples (loc = 0,
           scale = 1) in a transient GPU f64 scratch, then a DD-affine
           kernel applies the user's fp128 loc/scale on device. The
           entire pipeline stays VRAM-side. */
        double *z = NULL;
        vmalloc((void **)&z, (unsigned int)((size_t)n * sizeof(double)));
        if (z == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_truncated_normal_f64(z, n, 0.0, 1.0);

        double loc_hi, loc_lo, scale_hi, scale_lo;
        ndarray_fp128_split(spec->v.f128.loc, &loc_hi, &loc_lo);
        ndarray_fp128_split(spec->v.f128.scale, &scale_hi, &scale_lo);

        cuda_normal_dd_affine(z, (double *)rtn->data, n,
                               loc_hi, loc_lo, scale_hi, scale_lo);
        vfree(z);
        return rtn;
    }

    if (spec->kind == NDARRAY_NORMAL_KIND_UINT64) {
        /* Generate truncated standard-normal samples on GPU
           (`cuda_truncated_normal_f64` with loc=0, scale=1 gives values
           in [-2, 2]), then the shared `cuda_normal_u64_affine` kernel
           writes `loc + (uint64_t)((int64_t)(scale * z))` directly into
           the destination. Entire pipeline stays VRAM-side. */
        double *z = NULL;
        vmalloc((void **)&z, (unsigned int)((size_t)n * sizeof(double)));
        if (z == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_truncated_normal_f64(z, n, 0.0, 1.0);

        cuda_normal_u64_affine(z, (unsigned long long *)rtn->data, n,
                                (unsigned long long)spec->v.u64.loc,
                                (double)spec->v.u64.scale);
        vfree(z);
        return rtn;
    }
#endif

    /* Defensive: every reachable (device, kind) combination is handled
       above. If we get here something is mis-wired — free and bail. */
    NDArray_FREE(rtn);
    return NULL;
}

/* ───────────────── Poisson sampler ─────────────────────────────────────── */

/**
 * @brief Cached per-call constants for the CPU Poisson sampler.
 *
 * Selected once at the top of every `NDArray_Poisson` CPU branch and
 * reused across every element so the per-sample work is `O(1)` for
 * PTRS and `O(λ)` only for the Knuth branch where it's unavoidable.
 * Hoisting the constants out of the per-element loop is also the fix
 * for the legacy bug where `L = expf(-lam)` was recomputed in every
 * inner-loop iteration.
 */
typedef struct {
    double lam;
    int    use_ptrs;
    /* Knuth-multiplicative path (lam < 30): pre-exponential. */
    double L;
    /* PTRS path (Hörmann 1993, lam >= 30): rejection-region constants. */
    double smu;
    double b;
    double a;
    double inv_alpha;
    double v_r;
} ndarray_poisson_ctx;

/**
 * @brief Choose Knuth vs PTRS and precompute the per-call constants.
 *
 * Threshold: `lam < 30` keeps Knuth (expected `λ + 1` PRNG draws per
 * sample, cache-friendly). Beyond that, Knuth becomes wasteful and
 * `exp(-lam)` starts to lose precision; PTRS (Patchwork Rejection from
 * Triangular and Symmetric region — Hörmann 1993) takes a constant
 * `~1.3` PRNG draws per sample and stays numerically stable up to
 * lam ≈ 10^9. The threshold matches numpy's `random_poisson` switch
 * point (`distributions.c`).
 *
 * @param[out] ctx Receives the chosen branch + precomputed constants.
 * @param[in]  lam Distribution rate (λ); must be ≥ 0. lam = 0 is
 *                 special-cased by callers (always returns 0).
 */
static void ndarray_poisson_ctx_init(ndarray_poisson_ctx *ctx, double lam) {
    ctx->lam = lam;
    if (lam < 30.0) {
        ctx->use_ptrs = 0;
        ctx->L = exp(-lam);
    } else {
        ctx->use_ptrs  = 1;
        ctx->smu       = sqrt(lam);
        ctx->b         = 0.931 + 2.53 * ctx->smu;
        ctx->a         = -0.059 + 0.02483 * ctx->b;
        ctx->inv_alpha = 1.1239 + 1.1328 / (ctx->b - 3.4);
        ctx->v_r       = 0.9277 - 3.6224 / (ctx->b - 2.0);
    }
}

/**
 * @brief Knuth's multiplicative method for small λ.
 *
 * Draws k uniform `[0, 1)` samples and counts until the running
 * product drops below `L = exp(-λ)`. Expected runtime is `λ + 1`
 * PRNG calls per sample, so this branch is reserved for `λ < 30`.
 *
 * The uniform draw uses `rand() / (RAND_MAX + 1.0)` — the canonical
 * formulation that never yields 1.0 exactly (the legacy
 * `rand() / RAND_MAX` could, which was harmless here but inconsistent
 * with the rest of the rng plumbing).
 *
 * @param[in] L Precomputed `exp(-λ)`.
 * @return One Poisson sample (a non-negative integer as a double).
 */
static double ndarray_poisson_sample_knuth(double L) {
    double p = 1.0;
    int    k = 0;
    do {
        k++;
        double u = (double)rand() / ((double)RAND_MAX + 1.0);
        p *= u;
    } while (p > L);
    return (double)(k - 1);
}

/**
 * @brief Hörmann's PTRS rejection sampler for large λ.
 *
 * Reference: W. Hörmann, "The transformed rejection method for
 * generating Poisson random variables" (Insurance: Math. & Econ. 12,
 * 1993, p. 41). Same constants and dispatch as numpy's
 * `random_poisson_ptrs`. Per-call: 2 uniforms + `log` + `lgamma`; the
 * accept rate is `~78%`, so a sample averages ~1.3 PRNG draws regardless
 * of λ. Numerically stable from `λ = 30` up to `λ ≈ 10^9`.
 *
 * @param[in] ctx Initialised context (PTRS branch).
 * @return One Poisson sample (a non-negative integer as a double).
 */
static double ndarray_poisson_sample_ptrs(const ndarray_poisson_ctx *ctx) {
    double lam       = ctx->lam;
    double b         = ctx->b;
    double a         = ctx->a;
    double inv_alpha = ctx->inv_alpha;
    double v_r       = ctx->v_r;
    for (;;) {
        double u  = ((double)rand() / ((double)RAND_MAX + 1.0)) - 0.5;
        double v  = (double)rand() / ((double)RAND_MAX + 1.0);
        double us = 0.5 - fabs(u);
        double k  = floor((2.0 * a / us + b) * u + lam + 0.43);
        if (k < 0.0) continue;
        if (us >= 0.07 && v <= v_r) return k;
        if (us < 0.013 && v > us)   continue;
        if (log(v) + log(inv_alpha) - log(a / (us * us) + b) <=
            -lam + k * log(lam) - lgamma(k + 1.0)) {
            return k;
        }
    }
}

/**
 * @brief Draw one Poisson(λ) sample on the CPU.
 *
 * Dispatches to Knuth or PTRS via the cached context. For `λ == 0` the
 * sample is identically 0 (degenerate Poisson). Negative λ is rejected
 * upstream by the PHP entry point so this helper trusts its input.
 *
 * @param[in] ctx Initialised context.
 * @return One Poisson sample (a non-negative integer as a double).
 */
static double ndarray_poisson_sample(const ndarray_poisson_ctx *ctx) {
    if (ctx->lam <= 0.0) return 0.0;
    if (ctx->use_ptrs)   return ndarray_poisson_sample_ptrs(ctx);
    return ndarray_poisson_sample_knuth(ctx->L);
}

/**
 * @brief CPU fill: write @p n Poisson(@p lam) samples into @p data of @p type.
 *
 * Generates each sample as a `double` via `ndarray_poisson_sample` and
 * stores it through `ndarray_set_from_double` so every dtype gets its
 * dtype-correct quantisation. Covers every dtype except float128 (DD
 * widening) and uint64 (full 64-bit range) — both have dedicated
 * fillers below.
 *
 * @param[out] data Destination host buffer; ≥ `n * get_type_size(type)` bytes.
 * @param[in]  n    Element count.
 * @param[in]  type Canonical dtype string.
 * @param[in]  lam  Distribution rate (λ); ≥ 0.
 */
static void poisson_fill_cpu_double(char *data, long n, const char *type,
                                      double lam) {
    ndarray_poisson_ctx ctx;
    ndarray_poisson_ctx_init(&ctx, lam);
    for (long i = 0; i < n; i++) {
        double v = ndarray_poisson_sample(&ctx);
        ndarray_set_from_double(type, data, (size_t)i, v);
    }
}

/**
 * @brief CPU fill for the float128 dtype.
 *
 * Each Poisson sample is an integer (≤ ~10^9 in practice), so widening
 * to fp128 via `NDARRAY_FP128_FROM_D` is exact — the high word carries
 * the integer count, the low word is zero.
 *
 * @param[out] data Destination host buffer of `n * NDARRAY_FP128_SIZE` bytes.
 * @param[in]  n    Element count.
 * @param[in]  lam  Distribution rate (λ); ≥ 0.
 */
static void poisson_fill_cpu_fp128(char *data, long n, double lam) {
    ndarray_poisson_ctx ctx;
    ndarray_poisson_ctx_init(&ctx, lam);
    ndarray_fp128_t *p = (ndarray_fp128_t *)data;
    for (long i = 0; i < n; i++) {
        double v = ndarray_poisson_sample(&ctx);
        p[i] = NDARRAY_FP128_FROM_D(v);
    }
}

/**
 * @brief CPU fill for the uint64 dtype.
 *
 * Each Poisson sample is a non-negative integer (≤ ~10^9 in practice
 * — well inside the int64 range, so the conversion is exact). Stores
 * via a single `(uint64_t)` cast.
 *
 * @param[out] data Destination host buffer of `n * sizeof(uint64_t)` bytes.
 * @param[in]  n    Element count.
 * @param[in]  lam  Distribution rate (λ); ≥ 0.
 */
static void poisson_fill_cpu_uint64(char *data, long n, double lam) {
    ndarray_poisson_ctx ctx;
    ndarray_poisson_ctx_init(&ctx, lam);
    uint64_t *p = (uint64_t *)data;
    for (long i = 0; i < n; i++) {
        double v = ndarray_poisson_sample(&ctx);
        p[i] = (uint64_t)v;
    }
}

/**
 * @brief Build a Poisson-sample NDArray of the requested shape / dtype / device.
 *
 * For `device == NDARRAY_DEVICE_GPU` the destination buffer is allocated
 * directly in VRAM via `NDArray_Empty` (no host-side staging of the
 * result):
 *
 *  - `dtype == uint32`: `cuda_poisson_u32` writes directly into the
 *    destination. cuRAND's native Poisson output is `unsigned int`,
 *    so this is the cheapest path.
 *  - Other DOUBLE-kind dtypes: a transient GPU `uint32` scratch is
 *    allocated via `vmalloc`, filled by `cuda_poisson_u32`, and then
 *    `cuda_cast_u32_to_<dst>` quantises into the destination. Every
 *    Poisson sample is ≤ ~10^9 (cuRAND's documented bound on λ), so
 *    casting through u32 is precision-preserving for every dtype with
 *    a ≥ 32-bit mantissa; for narrower dtypes the cast saturates the
 *    same way numpy's `astype` does.
 *  - `dtype == float4` / `float8`: u32 → f32 → fp4 / fp8 (two-stage
 *    cast — the small-fp casts only have an f32 source on GPU).
 *  - `dtype == float128`: `cuda_cast_u32_to_dd` writes each integer
 *    sample as a `(double, 0.0)` DD pair, keeping the (hi, lo) layout
 *    bit-correct.
 *  - `dtype == uint64`: `cuda_cast_u32_to_u64` widens the u32 scratch
 *    to u64 (zero-extension) — no host stage, kernel writes directly
 *    into the VRAM destination.
 *
 * For `device == NDARRAY_DEVICE_CPU` every kind writes straight into
 * the destination host buffer via the per-kind fillers above.
 *
 * Pre-existing bugs fixed (legacy `NDArray_Poisson` was a single
 * `NDArray_Zeros + Knuth-loop-with-expf`):
 *  - **`expf(-lam)` underflowed to 0 for `lam ≥ 88`** → the Knuth
 *    inner loop never exited (`p > 0` always); first sample hung the
 *    process. Replaced with `exp(-lam)` (precise up to lam ≈ 700) and
 *    swapped to PTRS for `lam ≥ 30` so very large rates also work.
 *  - **`L = exp(-lam)` was recomputed in every inner-loop iteration**.
 *    Hoisted into the per-call context.
 *  - **`(float)rand() / (float)RAND_MAX` could return exactly 1.0** —
 *    harmless for Poisson (`p * 1.0 = p`) but inconsistent with the
 *    rest of the project's PRNG plumbing. Replaced with
 *    `rand() / (RAND_MAX + 1.0)`.
 *  - **Hardcoded float32 / CPU** — dtype and device were ignored.
 *  - **No GPU path at all** in the legacy implementation.
 *  - **`NDArray_Zeros` then immediately overwritten** — wasted memset;
 *    new path uses `NDArray_Empty` for uninitialised allocation.
 *
 * @param[in] lam    Distribution rate (λ); must be ≥ 0 (validated by
 *                   the PHP entry point).
 * @param[in] shape  Newly-allocated `int[ndim]`; ownership transfers
 *                   into the returned NDArray's `dimensions`.
 * @param[in] ndim   Number of dimensions; 0 yields a 0-D scalar.
 * @param[in] type   Canonical NDArray dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray on success, NULL on failure (Error in flight).
 */
NDArray*
NDArray_Poisson(double lam, int *shape, int ndim,
                 const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) efree(shape);
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    long n = (long) NDArray_NUMELEMENTS(rtn);
    if (n <= 0) {
        return rtn;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before requesting
           GPU. If they didn't, fail loudly instead of returning an
           NDArray with uninitialised on-device storage. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    if (device == NDARRAY_DEVICE_CPU) {
        if (!strcmp(type, "float128")) {
            poisson_fill_cpu_fp128((char *)rtn->data, n, lam);
        } else if (!strcmp(type, "uint64")) {
            poisson_fill_cpu_uint64((char *)rtn->data, n, lam);
        } else {
            poisson_fill_cpu_double((char *)rtn->data, n, type, lam);
        }
        return rtn;
    }

#ifdef HAVE_CUBLAS
    /* device == NDARRAY_DEVICE_GPU. Destination buffer is already in
       VRAM (NDArray_Empty allocated it via vmalloc/cudaMalloc). Each
       branch below writes into it without copying the result through
       host memory. */

    /* lam == 0 is degenerate (every sample is identically 0), and
       cuRAND's curandGeneratePoisson rejects lam=0 with a non-success
       status. Short-circuit with a single cudaMemset — every typed
       zero encoding is the all-bytes-zero pattern, so this works for
       every dtype including the (hi, lo) DD layout. */
    if (lam == 0.0) {
        cudaMemset(rtn->data, 0, (size_t)n * (size_t)elsize);
        return rtn;
    }

    if (!strcmp(type, "uint32")) {
        if (!cuda_poisson_u32((unsigned int *)rtn->data, n, lam)) {
            NDArray_FREE(rtn);
            zend_throw_error(NULL,
                "poisson: cuRAND rejected lam=%g on the GPU (lambda "
                "exceeds the generator's internal precision bound). "
                "Use the CPU device for very large lambda.", lam);
            return NULL;
        }
        return rtn;
    }

    /* All other dtypes: allocate a u32 scratch, fill via cuRAND, then
       cast into the destination. Scratch is freed before return. */
    unsigned int *scratch = NULL;
    vmalloc((void **)&scratch, (unsigned int)((size_t)n * sizeof(unsigned int)));
    if (scratch == NULL) {
        NDArray_FREE(rtn);
        return NULL;
    }
    if (!cuda_poisson_u32(scratch, n, lam)) {
        vfree(scratch);
        NDArray_FREE(rtn);
        zend_throw_error(NULL,
            "poisson: cuRAND rejected lam=%g on the GPU (lambda "
            "exceeds the generator's internal precision bound). "
            "Use the CPU device for very large lambda.", lam);
        return NULL;
    }

    if (!strcmp(type, "int8")) {
        cuda_cast_u32_to_i8(scratch, (int8_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "uint8")) {
        cuda_cast_u32_to_u8(scratch, (uint8_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "int16")) {
        cuda_cast_u32_to_i16(scratch, (int16_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "uint16")) {
        cuda_cast_u32_to_u16(scratch, (uint16_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "int32")) {
        cuda_cast_u32_to_i32(scratch, (int32_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "int64")) {
        cuda_cast_u32_to_i64(scratch, (int64_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "uint64")) {
        cuda_cast_u32_to_u64(scratch, (uint64_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "float16")) {
        cuda_cast_u32_to_f16(scratch, (uint16_t *)rtn->data, (int)n);
    } else if (!strcmp(type, "float32")) {
        cuda_cast_u32_to_f32(scratch, (float *)rtn->data, (int)n);
    } else if (!strcmp(type, "float64")) {
        cuda_cast_u32_to_f64(scratch, (double *)rtn->data, (int)n);
    } else if (!strcmp(type, "float128")) {
        cuda_cast_u32_to_dd(scratch, (double *)rtn->data, n);
    } else if (!strcmp(type, "float4") || !strcmp(type, "float8")) {
        /* fp4 / fp8 casts only have an f32 source on the GPU side, so
           we take a two-stage path: u32 → f32 scratch → fp4 / fp8. */
        float *scratch_f = NULL;
        vmalloc((void **)&scratch_f,
                (unsigned int)((size_t)n * sizeof(float)));
        if (scratch_f == NULL) {
            vfree(scratch);
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_cast_u32_to_f32(scratch, scratch_f, (int)n);
        if (!strcmp(type, "float4")) {
            cuda_cast_f32_to_fp4(scratch_f, (uint8_t *)rtn->data, (int)n);
        } else {
            cuda_cast_f32_to_fp8(scratch_f, (uint8_t *)rtn->data, (int)n);
        }
        vfree(scratch_f);
    } else {
        /* Unreachable: every supported dtype is covered above. */
        vfree(scratch);
        NDArray_FREE(rtn);
        return NULL;
    }
    vfree(scratch);
    return rtn;
#endif

    /* Defensive: every reachable (device, kind) combination is handled
       above. If we get here something is mis-wired — free and bail. */
    NDArray_FREE(rtn);
    return NULL;
}

/* ───────────────── Uniform sampler ─────────────────────────────────────── */

/**
 * @brief Sample one uniform `double` u ~ U([0, 1)).
 *
 * Returns `rand() / (RAND_MAX + 1.0)` — the canonical formulation that
 * never produces 1.0 exactly (which the legacy `(float)rand() /
 * (float)RAND_MAX` could, in violation of the documented `[low, high)`
 * contract). RAND_MAX is at least 32767 on POSIX; on glibc it's
 * `2^31 - 1`, giving ~31 bits of entropy — adequate for the float32 /
 * float64 paths. State is `static` inside the glibc `rand()` so it
 * mirrors the project's other CPU PRNG usage.
 *
 * @return One sample drawn from U([0, 1)).
 */
static double ndarray_uniform_sample(void) {
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

/**
 * @brief Map a canonical dtype string to its uniform-sampler arithmetic kind.
 *
 * `float128` returns FP128 so low/high stay at native fp128 precision;
 * `uint64` returns UINT64 so bounds past 2^53 stay exact; every other
 * dtype routes through `double`, which already covers each smaller
 * dtype's representable range.
 *
 * @param[in] type Canonical dtype string.
 * @return one of NDARRAY_UNIFORM_KIND_*.
 */
NDArrayUniformKind NDArray_UniformKindFor(const char *type) {
    if (!strcmp(type, "float128")) return NDARRAY_UNIFORM_KIND_FP128;
    if (!strcmp(type, "uint64"))   return NDARRAY_UNIFORM_KIND_UINT64;
    return NDARRAY_UNIFORM_KIND_DOUBLE;
}

/**
 * @brief CPU fill: write @p n U([low, high)) samples into @p data of @p type.
 *
 * Uses the `ndarray_uniform_sample` helper for the [0, 1) draw and
 * routes the per-element store through `ndarray_set_from_double` so
 * each dtype gets its dtype-correct quantisation (fp4 LUT pick,
 * integer truncation, etc.). Covers every dtype except float128 (DD
 * precision) and uint64 (full unsigned-64-bit range) — both have
 * dedicated fillers below.
 *
 * @param[out] data Destination host buffer; ≥ `n * get_type_size(type)` bytes.
 * @param[in]  n    Element count.
 * @param[in]  type Canonical dtype string.
 * @param[in]  low  Lower bound (inclusive).
 * @param[in]  high Upper bound (exclusive).
 */
static void uniform_fill_cpu_double(char *data, long n, const char *type,
                                     double low, double high) {
    double range = high - low;
    for (long i = 0; i < n; i++) {
        double v = low + ndarray_uniform_sample() * range;
        ndarray_set_from_double(type, data, (size_t)i, v);
    }
}

/**
 * @brief CPU fill for the float128 dtype.
 *
 * Computes each value as `value = low + u * (high - low)` in fp128
 * arithmetic so wide-range bounds (`'1e+200'` via string input) keep
 * full precision. The uniform draw itself is a double (53 bits) —
 * matches the CPU and GPU paths, where the underlying PRNG is fp64.
 *
 * @param[out] data Destination host buffer of `n * NDARRAY_FP128_SIZE` bytes.
 * @param[in]  n    Element count.
 * @param[in]  low  Lower bound as fp128.
 * @param[in]  high Upper bound as fp128.
 */
static void uniform_fill_cpu_fp128(char *data, long n,
                                    ndarray_fp128_t low,
                                    ndarray_fp128_t high) {
    ndarray_fp128_t *p = (ndarray_fp128_t *)data;
    ndarray_fp128_t range = NDARRAY_FP128_SUB(high, low);
    for (long i = 0; i < n; i++) {
        ndarray_fp128_t u = NDARRAY_FP128_FROM_D(ndarray_uniform_sample());
        p[i] = NDARRAY_FP128_ADD(low, NDARRAY_FP128_MUL(u, range));
    }
}

/**
 * @brief CPU fill for the uint64 dtype.
 *
 * Computes `value = low + (uint64_t)((high - low) * u)` keeping low /
 * high in `uint64_t` so bounds past 2^53 are bit-correct. The width
 * `(high - low)` uses unsigned modular arithmetic (matches numpy's
 * `astype(uint64)` semantics for the wraparound case `low > high`).
 *
 * @param[out] data Destination host buffer of `n * sizeof(uint64_t)` bytes.
 * @param[in]  n    Element count.
 * @param[in]  low  Lower bound as uint64.
 * @param[in]  high Upper bound as uint64.
 */
static void uniform_fill_cpu_uint64(char *data, long n,
                                     uint64_t low, uint64_t high) {
    uint64_t *p     = (uint64_t *)data;
    uint64_t width  = high - low;
    double   widthd = (double)width;
    for (long i = 0; i < n; i++) {
        double u = ndarray_uniform_sample();
        p[i] = low + (uint64_t)(widthd * u);
    }
}

/**
 * @brief Build a uniform-sample NDArray of the requested shape / dtype / device.
 *
 * Dispatches by the discriminated @p spec — `DOUBLE`, `FP128`, or
 * `UINT64` — and by @p device. For `device == NDARRAY_DEVICE_GPU` the
 * destination buffer is allocated directly in VRAM via `NDArray_Empty`
 * (no host-side staging of the result):
 *
 *  - `dtype == float32` / `float64`: cuRAND fills the destination
 *    in-place via `cuda_uniform_f32` / `cuda_uniform_f64`, which apply
 *    the `1 - u` reflection and the `[low, high)` affine on device.
 *  - Other dtypes with `DOUBLE` arithmetic: cuRAND fills a transient
 *    GPU `float32` scratch (allocated via `vmalloc`, freed before
 *    return), then a single `cuda_cast_f32_to_<dst>` quantises into the
 *    destination.
 *  - `int64` / dtypes with wider mantissas: cuRAND fills a transient
 *    `float64` scratch instead so values near 2^31..2^63 survive
 *    without f32-mantissa aliasing.
 *  - `dtype == float128`: cuRAND fills a GPU `float64` u-scratch, then a
 *    custom DD-affine kernel (`cuda_uniform_dd_affine`) computes
 *    `low + (1 - u) * (high - low)` in true double-double arithmetic
 *    on device, storing the (hi, lo) pair into the DD destination.
 *  - `dtype == uint64` with `UINT64` kind: `cuda_uniform_f64` fills a
 *    `[0, 1)` u-scratch, then `cuda_uniform_u64_affine` writes
 *    `low + (uint64_t)((high - low) * u)` directly into the
 *    destination — entire pipeline stays VRAM-side. The width
 *    `(double)(high - low)` is computed once on the host (same
 *    precision floor the CPU filler hits past 2^53).
 *
 * For `device == NDARRAY_DEVICE_CPU` every kind writes straight into
 * the destination host buffer via the per-kind fillers above.
 *
 * Pre-existing bugs fixed:
 *  - Legacy `NDArray_Uniform` hardcoded float32 / CPU regardless of caller
 *    request and ignored every other dtype.
 *  - Legacy fill called `NDArray_Zeros` then immediately overwrote the
 *    buffer — wasted memset; now routes through `NDArray_Empty` for an
 *    uninitialised allocation.
 *  - Legacy `(float)rand() / (float)RAND_MAX` could return exactly 1.0,
 *    violating the documented `[low, high)` contract by allowing the
 *    `high` endpoint. The new `ndarray_uniform_sample` helper uses
 *    `RAND_MAX + 1.0` so values are strictly in `[0, 1)` and the affine
 *    stays in `[low, high)`.
 *  - No GPU support at all in the legacy path.
 *
 * @param[in] spec   Discriminated (low, high) pair in the kind dictated
 *                   by @p type.
 * @param[in] shape  Newly-allocated `int[ndim]`; ownership transfers
 *                   into the returned NDArray's `dimensions`.
 * @param[in] ndim   Number of dimensions; 0 yields a 0-D scalar.
 * @param[in] type   Canonical NDArray dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray on success, NULL on failure (Error in flight).
 */
NDArray*
NDArray_Uniform(const NDArrayUniformSpec *spec, int *shape, int ndim,
                const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) efree(shape);
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    long n = (long) NDArray_NUMELEMENTS(rtn);
    if (n <= 0) {
        return rtn;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before requesting
           GPU. If they didn't, fail loudly instead of returning an
           NDArray with uninitialised on-device storage. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    if (device == NDARRAY_DEVICE_CPU) {
        switch (spec->kind) {
            case NDARRAY_UNIFORM_KIND_DOUBLE:
                uniform_fill_cpu_double((char *)rtn->data, n, type,
                                         spec->v.d.low, spec->v.d.high);
                break;
            case NDARRAY_UNIFORM_KIND_FP128:
                uniform_fill_cpu_fp128((char *)rtn->data, n,
                                        spec->v.f128.low, spec->v.f128.high);
                break;
            case NDARRAY_UNIFORM_KIND_UINT64:
                uniform_fill_cpu_uint64((char *)rtn->data, n,
                                         spec->v.u64.low, spec->v.u64.high);
                break;
        }
        return rtn;
    }

#ifdef HAVE_CUBLAS
    /* From here on: device == NDARRAY_DEVICE_GPU. The destination buffer
       lives in VRAM (NDArray_Empty allocated it via vmalloc/cudaMalloc).
       Each branch writes into it without ever copying the result through
       host memory. */
    if (spec->kind == NDARRAY_UNIFORM_KIND_DOUBLE) {
        double low  = spec->v.d.low;
        double high = spec->v.d.high;

        if (!strcmp(type, "float32")) {
            cuda_uniform_f32((float *)rtn->data, n, (float)low, (float)high);
            return rtn;
        }
        if (!strcmp(type, "float64")) {
            cuda_uniform_f64((double *)rtn->data, n, low, high);
            return rtn;
        }

        /* Every other dtype in the DOUBLE kind gets a transient GPU f32
           scratch filled by cuRAND, then a single cuda_cast_f32_to_<dst>
           pass quantises into the destination. Scratch lives only
           inside this call. */
        float *scratch = NULL;
        vmalloc((void **)&scratch, (unsigned int)((size_t)n * sizeof(float)));
        if (scratch == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_uniform_f32(scratch, n, (float)low, (float)high);

        if (!strcmp(type, "float4")) {
            cuda_cast_f32_to_fp4(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "float8")) {
            cuda_cast_f32_to_fp8(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "float16")) {
            cuda_cast_f32_to_f16(scratch, (uint16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int8")) {
            cuda_cast_f32_to_i8(scratch, (int8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint8")) {
            cuda_cast_f32_to_u8(scratch, (uint8_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int16")) {
            cuda_cast_f32_to_i16(scratch, (int16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint16")) {
            cuda_cast_f32_to_u16(scratch, (uint16_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int32")) {
            cuda_cast_f32_to_i32(scratch, (int32_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "uint32")) {
            cuda_cast_f32_to_u32(scratch, (uint32_t *)rtn->data, (int)n);
        } else if (!strcmp(type, "int64")) {
            /* int64 has range past float32's 24-bit mantissa. Promote
               the scratch to fp64 before casting so values near 2^31..2^63
               survive without aliasing. */
            double *scratch64 = NULL;
            vmalloc((void **)&scratch64,
                    (unsigned int)((size_t)n * sizeof(double)));
            if (scratch64 != NULL) {
                cuda_uniform_f64(scratch64, n, low, high);
                cuda_cast_f64_to_i64(scratch64, (int64_t *)rtn->data, (int)n);
                vfree(scratch64);
            }
        } else {
            /* Should be unreachable: NDArray_UniformKindFor only returns
               DOUBLE for these dtypes. */
            vfree(scratch);
            NDArray_FREE(rtn);
            return NULL;
        }
        vfree(scratch);
        return rtn;
    }

    if (spec->kind == NDARRAY_UNIFORM_KIND_FP128) {
        /* Generate fp64 u-samples on GPU, then a DD-affine kernel writes
           the result DD pair into the destination — entire pipeline
           stays VRAM-side; low/range go to the device as scalar args. */
        double *u = NULL;
        vmalloc((void **)&u, (unsigned int)((size_t)n * sizeof(double)));
        if (u == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        /* Use the raw cuRAND draw (without the host-side affine) so we
           keep all of the affine precision in the device-side DD math.
           A direct curandGenerateUniformDouble would also work, but the
           public wrapper applies an fp64 affine internally — we want the
           DD kernel to be the only path that touches range arithmetic.
           So we fill with `low = 0`, `high = 1` to get a passthrough
           `[0, 1)` stream, then the DD kernel applies the real affine. */
        cuda_uniform_f64(u, n, 0.0, 1.0);

        ndarray_fp128_t range_f = NDARRAY_FP128_SUB(spec->v.f128.high,
                                                    spec->v.f128.low);
        double low_hi, low_lo, range_hi, range_lo;
        ndarray_fp128_split(spec->v.f128.low, &low_hi, &low_lo);
        ndarray_fp128_split(range_f,          &range_hi, &range_lo);

        cuda_uniform_dd_affine(u, (double *)rtn->data, n,
                                low_hi, low_lo, range_hi, range_lo);
        vfree(u);
        return rtn;
    }

    if (spec->kind == NDARRAY_UNIFORM_KIND_UINT64) {
        /* Generate fp64 [0, 1) samples on GPU, then a u64 affine kernel
           writes `low + (uint64_t)(width * u)` directly into the
           destination — entire pipeline stays VRAM-side. `width` is
           computed on the host because the (high - low) subtraction in
           uint64 is unsigned-modular (matches the CPU filler) and the
           result is then cast once to double for the affine. */
        double *u = NULL;
        vmalloc((void **)&u, (unsigned int)((size_t)n * sizeof(double)));
        if (u == NULL) {
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_uniform_f64(u, n, 0.0, 1.0);

        uint64_t width  = spec->v.u64.high - spec->v.u64.low;
        double   widthd = (double)width;
        cuda_uniform_u64_affine(u, (unsigned long long *)rtn->data, n,
                                 (unsigned long long)spec->v.u64.low,
                                 widthd);
        vfree(u);
        return rtn;
    }
#endif

    /* Defensive: every reachable (device, kind) combination is handled
       above. If we get here something is mis-wired — free and bail. */
    NDArray_FREE(rtn);
    return NULL;
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

/* ───────────────── Binomial sampler ────────────────────────────────────── */

/**
 * @brief Draw one Binomial(@p n, @p p) sample on the CPU.
 *
 * Direct Bernoulli method: count successes across @p n independent
 * trials with success probability @p p. Each uniform draw uses
 * `rand() / (RAND_MAX + 1.0)` so values land strictly in `[0, 1)`
 * (the legacy `(float)rand() / (float)RAND_MAX` could yield 1.0,
 * which would cause a `p = 1.0` request to miss a success).
 *
 * Cost is `O(n)` per call — same complexity as the cuRAND-based GPU
 * kernel. For very large @p n a BTPE-style algorithm would scale
 * better; the direct method stays numerically exact regardless, so
 * the distribution contract holds for any `n`.
 *
 * @param[in] n Number of Bernoulli trials.
 * @param[in] p Per-trial success probability in `[0, 1]`.
 * @return One Binomial sample (a non-negative integer as a double).
 */
static double ndarray_binomial_sample(int n, double p) {
    int successes = 0;
    for (int j = 0; j < n; j++) {
        double u = (double)rand() / ((double)RAND_MAX + 1.0);
        if (u < p) successes++;
    }
    return (double)successes;
}

/**
 * @brief CPU fill: write @p total Binomial(@p n, @p p) samples into @p data.
 *
 * Generates each sample via `ndarray_binomial_sample` and stores it
 * through `ndarray_set_from_double` so every dtype gets its
 * dtype-correct quantisation. Covers every dtype except float128 (DD
 * widening) and uint64 (full 64-bit range) — both have dedicated
 * fillers below.
 *
 * @param[out] data  Destination host buffer; ≥ `total * get_type_size(type)` bytes.
 * @param[in]  total Element count.
 * @param[in]  type  Canonical dtype string.
 * @param[in]  n     Trial count.
 * @param[in]  p     Success probability.
 */
static void binomial_fill_cpu_double(char *data, long total,
                                       const char *type, int n, double p) {
    for (long i = 0; i < total; i++) {
        double v = ndarray_binomial_sample(n, p);
        ndarray_set_from_double(type, data, (size_t)i, v);
    }
}

/**
 * @brief CPU fill for the float128 dtype.
 *
 * Each Binomial sample is an integer in `[0, n]` (so ≤ `INT_MAX`),
 * which widens to fp128 via `NDARRAY_FP128_FROM_D` losslessly.
 *
 * @param[out] data  Destination host buffer of `total * NDARRAY_FP128_SIZE` bytes.
 * @param[in]  total Element count.
 * @param[in]  n     Trial count.
 * @param[in]  p     Success probability.
 */
static void binomial_fill_cpu_fp128(char *data, long total, int n, double p) {
    ndarray_fp128_t *ptr = (ndarray_fp128_t *)data;
    for (long i = 0; i < total; i++) {
        double v = ndarray_binomial_sample(n, p);
        ptr[i] = NDARRAY_FP128_FROM_D(v);
    }
}

/**
 * @brief CPU fill for the uint64 dtype.
 *
 * Each Binomial sample is a non-negative integer in `[0, n]`. Stores
 * via a single `(uint64_t)` cast.
 *
 * @param[out] data  Destination host buffer of `total * sizeof(uint64_t)` bytes.
 * @param[in]  total Element count.
 * @param[in]  n     Trial count.
 * @param[in]  p     Success probability.
 */
static void binomial_fill_cpu_uint64(char *data, long total, int n, double p) {
    uint64_t *ptr = (uint64_t *)data;
    for (long i = 0; i < total; i++) {
        double v = ndarray_binomial_sample(n, p);
        ptr[i] = (uint64_t)v;
    }
}

/**
 * @brief Build a Binomial-sample NDArray of the requested shape / dtype / device.
 *
 * For `device == NDARRAY_DEVICE_GPU` the destination buffer is allocated
 * directly in VRAM via `NDArray_Empty` (no host-side staging of the
 * result):
 *
 *  - `dtype == uint32`: `cuda_binomial_u32` writes directly into the
 *    destination. The per-thread cuRAND state generates @p n uniforms
 *    per output slot and counts the successes.
 *  - Other dtypes: a transient GPU `uint32` scratch is allocated via
 *    `vmalloc`, filled by `cuda_binomial_u32`, and then
 *    `cuda_cast_u32_to_<dst>` quantises into the destination. Same
 *    plumbing as `NDArray_Poisson`.
 *  - `dtype == float128`: scratch → `cuda_cast_u32_to_dd` widens each
 *    count into a `(double, 0.0)` DD pair, keeping the (hi, lo)
 *    layout bit-correct.
 *  - `dtype == float4` / `float8`: u32 → f32 → fp4 / fp8 (two-stage —
 *    the small-fp casts only have an f32 source on GPU).
 *
 * For `device == NDARRAY_DEVICE_CPU` every dtype writes straight into
 * the destination via the per-kind fillers above.
 *
 * `n == 0` is degenerate (every sample is identically 0). The
 * dispatcher short-circuits with `cudaMemset(0)` (GPU) or a typed
 * `memset(0)` is implicit through `ndarray_set_from_double(..., 0.0)`
 * (CPU) — both yield the all-bytes-zero pattern which is the typed
 * encoding of 0 for every supported dtype.
 *
 * Pre-existing bugs fixed (legacy `NDArray_Binomial`):
 *  - **Hardcoded float32 / CPU** — dtype/device args were absent
 *    entirely.
 *  - **`(float)rand() / (float)RAND_MAX` could return exactly 1.0**,
 *    which under the `random_value < p` test missed a success when
 *    `p == 1.0`. Fixed: `rand() / (RAND_MAX + 1.0)` for strict `[0, 1)`.
 *  - **`NDArray_Zeros` then immediately overwritten** — wasted memset;
 *    new path uses `NDArray_Empty` for uninitialised allocation.
 *  - **No GPU path at all** in the legacy implementation.
 *  - **No `n < 0` / `p ∉ [0, 1]` validation** — the algorithm produced
 *    silent garbage for out-of-range inputs. Validated at the PHP
 *    entry point.
 *  - **`total_elements` was `int`** — overflowed for shapes whose
 *    product exceeded `INT_MAX`. The new dispatcher routes through
 *    `NDArray_Empty` whose element count is `long`.
 *
 * @param[in] shape  Newly-allocated `int[ndim]`; ownership transfers
 *                   into the returned NDArray's `dimensions`.
 * @param[in] ndim   Number of dimensions; 0 yields a 0-D scalar.
 * @param[in] n      Number of Bernoulli trials per sample; ≥ 0.
 * @param[in] p      Per-trial success probability in `[0, 1]`.
 * @param[in] type   Canonical NDArray dtype string.
 * @param[in] device NDARRAY_DEVICE_CPU or NDARRAY_DEVICE_GPU.
 * @return New NDArray on success, NULL on failure (Error in flight).
 */
NDArray*
NDArray_Binomial(int *shape, int ndim, int n, float p,
                  const char *type, int device) {
    int elsize = get_type_size(type);
    if (elsize == 0) {
        if (shape != NULL) efree(shape);
        return NULL;
    }

    NDArray *rtn = NDArray_Empty(shape, ndim, type, device);
    if (rtn == NULL) {
        return NULL;
    }
    long total = (long) NDArray_NUMELEMENTS(rtn);
    if (total <= 0) {
        return rtn;
    }

#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        /* Defensive: callers must gate on HAVE_CUBLAS before requesting
           GPU. If they didn't, fail loudly instead of returning an
           NDArray with uninitialised on-device storage. */
        NDArray_FREE(rtn);
        return NULL;
    }
#endif

    if (device == NDARRAY_DEVICE_CPU) {
        if (!strcmp(type, "float128")) {
            binomial_fill_cpu_fp128((char *)rtn->data, total, n, (double)p);
        } else if (!strcmp(type, "uint64")) {
            binomial_fill_cpu_uint64((char *)rtn->data, total, n, (double)p);
        } else {
            binomial_fill_cpu_double((char *)rtn->data, total, type, n,
                                       (double)p);
        }
        return rtn;
    }

#ifdef HAVE_CUBLAS
    /* device == NDARRAY_DEVICE_GPU. Destination buffer is already in
       VRAM (NDArray_Empty allocated it via vmalloc/cudaMalloc). Each
       branch below writes into it without copying the result through
       host memory. */

    /* n == 0 (or p == 0) — every sample is identically 0. Short-circuit
       with cudaMemset so we skip the per-thread cuRAND init for the
       degenerate case. The zero-byte pattern is the typed encoding of
       0 for every supported dtype, including the DD layout. */
    if (n == 0 || p == 0.0f) {
        cudaMemset(rtn->data, 0, (size_t)total * (size_t)elsize);
        return rtn;
    }

    if (!strcmp(type, "uint32")) {
        cuda_binomial_u32((unsigned int *)rtn->data, total, n, p);
        return rtn;
    }

    /* All other dtypes: allocate a u32 scratch, fill via the per-thread
       Bernoulli kernel, then cast into the destination. */
    unsigned int *scratch = NULL;
    vmalloc((void **)&scratch,
            (unsigned int)((size_t)total * sizeof(unsigned int)));
    if (scratch == NULL) {
        NDArray_FREE(rtn);
        return NULL;
    }
    cuda_binomial_u32(scratch, total, n, p);

    if (!strcmp(type, "int8")) {
        cuda_cast_u32_to_i8(scratch, (int8_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "uint8")) {
        cuda_cast_u32_to_u8(scratch, (uint8_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "int16")) {
        cuda_cast_u32_to_i16(scratch, (int16_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "uint16")) {
        cuda_cast_u32_to_u16(scratch, (uint16_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "int32")) {
        cuda_cast_u32_to_i32(scratch, (int32_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "int64")) {
        cuda_cast_u32_to_i64(scratch, (int64_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "uint64")) {
        cuda_cast_u32_to_u64(scratch, (uint64_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "float16")) {
        cuda_cast_u32_to_f16(scratch, (uint16_t *)rtn->data, (int)total);
    } else if (!strcmp(type, "float32")) {
        cuda_cast_u32_to_f32(scratch, (float *)rtn->data, (int)total);
    } else if (!strcmp(type, "float64")) {
        cuda_cast_u32_to_f64(scratch, (double *)rtn->data, (int)total);
    } else if (!strcmp(type, "float128")) {
        cuda_cast_u32_to_dd(scratch, (double *)rtn->data, total);
    } else if (!strcmp(type, "float4") || !strcmp(type, "float8")) {
        /* fp4 / fp8 casts only have an f32 source on the GPU side, so
           we take a two-stage path: u32 → f32 scratch → fp4 / fp8. */
        float *scratch_f = NULL;
        vmalloc((void **)&scratch_f,
                (unsigned int)((size_t)total * sizeof(float)));
        if (scratch_f == NULL) {
            vfree(scratch);
            NDArray_FREE(rtn);
            return NULL;
        }
        cuda_cast_u32_to_f32(scratch, scratch_f, (int)total);
        if (!strcmp(type, "float4")) {
            cuda_cast_f32_to_fp4(scratch_f, (uint8_t *)rtn->data, (int)total);
        } else {
            cuda_cast_f32_to_fp8(scratch_f, (uint8_t *)rtn->data, (int)total);
        }
        vfree(scratch_f);
    } else {
        /* Unreachable: every supported dtype is covered above. */
        vfree(scratch);
        NDArray_FREE(rtn);
        return NULL;
    }
    vfree(scratch);
    return rtn;
#endif

    /* Defensive: every reachable (device, kind) combination is handled
       above. If we get here something is mis-wired — free and bail. */
    NDArray_FREE(rtn);
    return NULL;
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
