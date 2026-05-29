#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include <string.h>
#include <math.h>
#include <float.h>
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

/**
 * @brief Identity-value enumeration for the four reduction operations.
 *
 * Shared by the CPU walk in `ndarray_reduce_cpu` and (on CUDA builds) the
 * GPU dispatcher; declared at file scope so it remains visible whether or
 * not `HAVE_CUBLAS` is defined. macOS clang errors on the forward-only
 * declaration that ifdef-guarding produces.
 */
enum ndarray_reduce_op { ND_RED_SUM, ND_RED_PROD, ND_RED_MIN, ND_RED_MAX };

#ifdef HAVE_CUBLAS
/**
 * @brief Return the identity value for reduction @p op as a double.
 *
 * sum = 0, prod = 1, min = +DBL_MAX, max = -DBL_MAX. Used to seed the
 * GPU accumulator so the first atomic merge produces the correct result.
 */
static inline double ndarray_reduce_identity(enum ndarray_reduce_op op) {
    switch (op) {
        case ND_RED_SUM:  return 0.0;
        case ND_RED_PROD: return 1.0;
        case ND_RED_MIN:  return DBL_MAX;
        case ND_RED_MAX:  return -DBL_MAX;
    }
    return 0.0;
}

/**
 * @brief Dispatch a single-axis-free reduction to the per-dtype GPU
 *        kernel and return the result as a double.
 *
 * Every dtype maps to a dedicated `cuda_reduce_<op>_<tag>` launch (see
 * `src/ndmath/cuda/cuda_math.h`). The accumulator is a single
 * GPU-resident double slot seeded with the operation's identity before
 * launch; the kernel atomically merges per-element values into that
 * slot, then we cudaMemcpy the final value back to host and free the
 * slot. No D2H of the input data ever happens — the entire reduction
 * runs on the device.
 *
 * @param[in] a   GPU-resident input NDArray.
 * @param[in] op  Which reduction to perform.
 * @return Reduction result as `double`; on dispatch failure (e.g.
 *         allocation), throws a PHP error and returns 0.0.
 */
static double
ndarray_reduce_dispatch_gpu(NDArray *a, enum ndarray_reduce_op op) {
    long n = NDArray_NUMELEMENTS(a);
    const char *t = NDArray_TYPE(a);
    const void *src = (const void *) NDArray_DATA(a);
    double *d_acc = NULL;
    vmalloc((void **) &d_acc, sizeof(double));
    if (d_acc == NULL) {
        zend_throw_error(NULL, "Failed to allocate GPU reduction accumulator");
        return 0.0;
    }
    double seed = ndarray_reduce_identity(op);
    cudaError_t cerr = cudaMemcpy(d_acc, &seed, sizeof(double), cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        vfree(d_acc);
        zend_throw_error(NULL, "GPU reduction seed copy failed: %s",
                         cudaGetErrorString(cerr));
        return 0.0;
    }

    int dispatched = 0;
#define DISPATCH_PAIR(DT_STR, TAG, T)                                              \
    if (!dispatched && is_type(t, DT_STR)) {                                        \
        switch (op) {                                                               \
            case ND_RED_SUM:  cuda_reduce_sum_##TAG ((const T *) src, d_acc, (int) n); break; \
            case ND_RED_PROD: cuda_reduce_prod_##TAG((const T *) src, d_acc, (int) n); break; \
            case ND_RED_MIN:  cuda_reduce_min_##TAG ((const T *) src, d_acc, (int) n); break; \
            case ND_RED_MAX:  cuda_reduce_max_##TAG ((const T *) src, d_acc, (int) n); break; \
        }                                                                           \
        dispatched = 1;                                                             \
    }
    DISPATCH_PAIR("int8",     i8,  int8_t)
    DISPATCH_PAIR("uint8",    u8,  uint8_t)
    DISPATCH_PAIR("int16",    i16, int16_t)
    DISPATCH_PAIR("uint16",   u16, uint16_t)
    DISPATCH_PAIR("int32",    i32, int32_t)
    DISPATCH_PAIR("uint32",   u32, uint32_t)
    DISPATCH_PAIR("int64",    i64, int64_t)
    DISPATCH_PAIR("uint64",   u64, uint64_t)
    DISPATCH_PAIR("float32",  f32, float)
    DISPATCH_PAIR("float64",  f64, double)
    DISPATCH_PAIR("float16",  f16, uint16_t)
    DISPATCH_PAIR("float4",   fp4, uint8_t)
    DISPATCH_PAIR("float8",   fp8, uint8_t)
    DISPATCH_PAIR("float128", dd,  double)
#undef DISPATCH_PAIR

    if (!dispatched) {
        /* No kernel ran — the result would silently be the identity value.
           Surface this as a hard error so the caller knows the dtype is
           unsupported on the GPU path. The accumulator is still vfree'd
           below to keep VCHECK happy. */
        vfree(d_acc);
        zend_throw_error(NULL,
                         "GPU reduction is not implemented for dtype \"%s\"", t);
        return 0.0;
    }
    /* After the kernel launch, surface any configuration / launch error so
       it doesn't silently appear inside an unrelated call later. The result
       D2H below is a blocking cudaMemcpy on the same default stream, which
       also synchronises and flushes any pending async error. */
    cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) {
        vfree(d_acc);
        zend_throw_error(NULL, "GPU reduction kernel launch failed: %s",
                         cudaGetErrorString(cerr));
        return 0.0;
    }
    double result = seed;
    cerr = cudaMemcpy(&result, d_acc, sizeof(double), cudaMemcpyDeviceToHost);
    vfree(d_acc);
    if (cerr != cudaSuccess) {
        zend_throw_error(NULL, "GPU reduction result copy failed: %s",
                         cudaGetErrorString(cerr));
        return 0.0;
    }
    return result;
}
#endif /* HAVE_CUBLAS */

/**
 * @brief CPU-side generic reduction over every supported dtype.
 *
 * Walks @p a's host buffer element by element via
 * `ndarray_element_to_double` and accumulates in a `double`. Used only
 * when @p a is CPU-resident; GPU inputs route through
 * `ndarray_reduce_dispatch_gpu` so the entire reduction stays on device.
 *
 * @param[in] a Source NDArray, must be CPU-resident.
 * @param[in] op Which reduction to perform.
 * @return Reduction result; throws on empty min / max.
 */
static double
ndarray_reduce_cpu(NDArray *a, enum ndarray_reduce_op op) {
    long n = NDArray_NUMELEMENTS(a);
    const char *t = NDArray_TYPE(a);
    const char *src = (const char *) NDArray_DATA(a);
    switch (op) {
        case ND_RED_SUM: {
            double s = 0.0;
            for (long i = 0; i < n; i++) {
                s += ndarray_element_to_double(t, src, (size_t) i);
            }
            return s;
        }
        case ND_RED_PROD: {
            double p = 1.0;
            for (long i = 0; i < n; i++) {
                p *= ndarray_element_to_double(t, src, (size_t) i);
            }
            return p;
        }
        case ND_RED_MIN:
        case ND_RED_MAX: {
            double cur = ndarray_element_to_double(t, src, 0);
            for (long i = 1; i < n; i++) {
                double v = ndarray_element_to_double(t, src, (size_t) i);
                if (isnan(v)) return v;
                if (op == ND_RED_MIN ? (v < cur) : (v > cur)) cur = v;
            }
            return cur;
        }
    }
    return 0.0;
}

/**
 * @brief Sum of all elements as a double, dispatched by dtype and device.
 *
 * GPU inputs are reduced entirely on the device via the per-dtype
 * `cuda_reduce_sum_<tag>` kernels; no host staging. CPU inputs walk
 * the buffer with `ndarray_element_to_double`. Accumulator is `double`
 * regardless of source dtype.
 *
 * Empty input (n == 0) returns `0.0`, matching NumPy's sum.identity.
 *
 * @param[in] a Input NDArray (any dtype, any device).
 * @return Sum as `double`.
 */
double
NDArray_Reduce_Sum(NDArray *a) {
    long n = NDArray_NUMELEMENTS(a);
    if (n <= 0) return 0.0;
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        return ndarray_reduce_dispatch_gpu(a, ND_RED_SUM);
    }
#endif
    return ndarray_reduce_cpu(a, ND_RED_SUM);
}

/**
 * @brief Product of all elements as a double, dispatched by dtype and device.
 *
 * GPU inputs use `cuda_reduce_prod_<tag>`; CPU inputs walk the host
 * buffer. Empty input returns `1.0` (prod.identity).
 *
 * @param[in] a Input NDArray (any dtype, any device).
 * @return Product as `double`.
 */
double
NDArray_Reduce_Prod(NDArray *a) {
    long n = NDArray_NUMELEMENTS(a);
    if (n <= 0) return 1.0;
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        return ndarray_reduce_dispatch_gpu(a, ND_RED_PROD);
    }
#endif
    return ndarray_reduce_cpu(a, ND_RED_PROD);
}

/**
 * @brief Minimum element as a double, dispatched by dtype and device.
 *
 * GPU inputs use `cuda_reduce_min_<tag>`; CPU inputs walk the host
 * buffer. NaN propagates: a single NaN element forces a NaN result.
 *
 * @param[in] a Input NDArray (any dtype, any device).
 * @return Minimum value; throws on empty input.
 */
double
NDArray_Reduce_Min(NDArray *a) {
    long n = NDArray_NUMELEMENTS(a);
    if (n <= 0) {
        zend_throw_error(NULL, "zero-size array to reduction operation min");
        return 0.0;
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        return ndarray_reduce_dispatch_gpu(a, ND_RED_MIN);
    }
#endif
    return ndarray_reduce_cpu(a, ND_RED_MIN);
}

/**
 * @brief Maximum element as a double, dispatched by dtype and device.
 *
 * GPU inputs use `cuda_reduce_max_<tag>`; CPU inputs walk the host
 * buffer. NaN propagates as in `NDArray_Reduce_Min`.
 *
 * @param[in] a Input NDArray (any dtype, any device).
 * @return Maximum value; throws on empty input.
 */
double
NDArray_Reduce_Max(NDArray *a) {
    long n = NDArray_NUMELEMENTS(a);
    if (n <= 0) {
        zend_throw_error(NULL, "zero-size array to reduction operation max");
        return 0.0;
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        return ndarray_reduce_dispatch_gpu(a, ND_RED_MAX);
    }
#endif
    return ndarray_reduce_cpu(a, ND_RED_MAX);
}

/**
 * @brief Arithmetic mean of all elements as a double, dispatched by dtype.
 *
 * Computed as `NDArray_Reduce_Sum(a) / n`. Empty input returns `NaN`,
 * matching NumPy's mean on a zero-size array.
 *
 * @param[in] a Input NDArray (any dtype, any device).
 * @return Mean as `double`.
 */
double
NDArray_Reduce_Mean(NDArray *a) {
    long n = NDArray_NUMELEMENTS(a);
    if (n <= 0) return NAN;
    return NDArray_Reduce_Sum(a) / (double) n;
}

/**
 * @brief Pick the result dtype for a sum / prod reduction over @p input_dt.
 *
 * Default widening (PyTorch parity):
 *  - signed   integer dtypes (int8..int64)   widen to `int64`,
 *  - unsigned integer dtypes (uint8..uint64) widen to `uint64`,
 *  - narrow   floating dtypes (float4..float16) widen to `float32` so the
 *    sum / prod doesn't saturate at the dtype's tiny representable range
 *    (`float4` max = 6, `float8 E4M3` max = 240, `float16` max ≈ 65504),
 *  - `float32` / `float64` / `float128` keep the source dtype.
 *
 * This is the only place where the reduction's output dtype differs from
 * the input — `NDArray_ScalarToZval` then picks the correct PHP-scalar
 * encoding for the widened type (`string` for `uint64`/`float128`,
 * `int` for the remaining integer dtypes, `float` otherwise).
 *
 * @param[in] input_dt Source NDArray dtype.
 * @return Canonical dtype string for the reduction's output.
 */
static const char *
ndarray_reduce_result_dtype(const char *input_dt) {
    if (!strcmp(input_dt, "int8")   || !strcmp(input_dt, "int16")  ||
        !strcmp(input_dt, "int32")  || !strcmp(input_dt, "int64"))  return "int64";
    if (!strcmp(input_dt, "uint8")  || !strcmp(input_dt, "uint16") ||
        !strcmp(input_dt, "uint32") || !strcmp(input_dt, "uint64")) return "uint64";
    if (!strcmp(input_dt, "float4") || !strcmp(input_dt, "float8")  ||
        !strcmp(input_dt, "float16")) return "float32";
    return input_dt;
}

/**
 * @brief Read element @p idx of @p data (dtype @p t) as `int64_t`.
 *
 * Used by the int64-accumulator path of `sum` / `prod` so narrow signed and
 * unsigned int inputs can be safely widened (each element exactly representable
 * as `int64_t`; the accumulator then wraps on overflow per C signed-integer
 * semantics).
 *
 * @param[in] t    Dtype of the source buffer.
 * @param[in] data Source buffer pointer.
 * @param[in] idx  Element index.
 * @return Element value widened to `int64_t`.
 */
static int64_t
ndarray_element_to_int64(const char *t, const char *data, size_t idx) {
    if (!strcmp(t, "int8"))   return (int64_t)(int8_t)((const uint8_t *)data)[idx];
    if (!strcmp(t, "uint8"))  return (int64_t)((const uint8_t *)data)[idx];
    if (!strcmp(t, "int16"))  { int16_t v;  memcpy(&v, (const int16_t  *)data + idx, 2); return (int64_t)v; }
    if (!strcmp(t, "uint16")) { uint16_t v; memcpy(&v, (const uint16_t *)data + idx, 2); return (int64_t)v; }
    if (!strcmp(t, "int32"))  { int32_t v;  memcpy(&v, (const int32_t  *)data + idx, 4); return (int64_t)v; }
    if (!strcmp(t, "uint32")) { uint32_t v; memcpy(&v, (const uint32_t *)data + idx, 4); return (int64_t)v; }
    if (!strcmp(t, "int64"))  { int64_t v;  memcpy(&v, (const int64_t  *)data + idx, 8); return v; }
    if (!strcmp(t, "uint64")) { uint64_t v; memcpy(&v, (const uint64_t *)data + idx, 8); return (int64_t)v; }
    /* Non-integer fallback — round to nearest. */
    return (int64_t)ndarray_element_to_double(t, data, idx);
}

/**
 * @brief Read element @p idx of @p data (dtype @p t) as `uint64_t`.
 *
 * Companion to `ndarray_element_to_int64` for the uint64-accumulator path.
 * Signed inputs are widened through `int64_t` first so negative values
 * sign-extend before reinterpretation, matching the two's-complement
 * behaviour of an explicit `(uint64_t)(int64_t)v` cast.
 *
 * @param[in] t    Dtype of the source buffer.
 * @param[in] data Source buffer pointer.
 * @param[in] idx  Element index.
 * @return Element value widened to `uint64_t`.
 */
static uint64_t
ndarray_element_to_uint64(const char *t, const char *data, size_t idx) {
    if (!strcmp(t, "uint8"))  return (uint64_t)((const uint8_t *)data)[idx];
    if (!strcmp(t, "uint16")) { uint16_t v; memcpy(&v, (const uint16_t *)data + idx, 2); return (uint64_t)v; }
    if (!strcmp(t, "uint32")) { uint32_t v; memcpy(&v, (const uint32_t *)data + idx, 4); return (uint64_t)v; }
    if (!strcmp(t, "uint64")) { uint64_t v; memcpy(&v, (const uint64_t *)data + idx, 8); return v; }
    return (uint64_t)ndarray_element_to_int64(t, data, idx);
}

/**
 * @brief Read element @p idx of @p data (dtype @p t) as `ndarray_fp128_t`.
 *
 * Loss-free for `float128` inputs (direct memcpy of the 16-byte storage);
 * for every other dtype the element first widens through `double` (exact
 * for `float4..float64` and every integer dtype except `uint64` /
 * `int64` past the double mantissa). The fp128 accumulator absorbs the
 * exact double, which is far wider than the source dtype's range.
 *
 * @param[in] t    Dtype of the source buffer.
 * @param[in] data Source buffer pointer.
 * @param[in] idx  Element index.
 * @return Element widened to `ndarray_fp128_t`.
 */
static ndarray_fp128_t
ndarray_element_to_fp128(const char *t, const char *data, size_t idx) {
    if (!strcmp(t, "float128")) {
        ndarray_fp128_t v;
        memcpy(&v, (const ndarray_fp128_t *)data + idx, NDARRAY_FP128_SIZE);
        return v;
    }
    return NDARRAY_FP128_FROM_D(ndarray_element_to_double(t, data, idx));
}

/**
 * @brief CPU sum reduction with native-precision accumulator selected by
 *        @p out_dt.
 *
 * Walks every element of @p src once and writes the final accumulator into
 * the first slot of @p out:
 *  - `int64`    → `int64_t` accumulator (signed-overflow wrap matches C),
 *  - `uint64`   → `uint64_t` accumulator (modular two's-complement wrap),
 *  - `float128` → `ndarray_fp128_t` accumulator (native fp128 add),
 *  - everything else → `double` accumulator stored via
 *    `ndarray_set_from_double` (rounds to the target dtype's precision).
 *
 * @param[in]  src    Source NDArray, CPU resident.
 * @param[out] out    Destination NDArray (0-D / 1-element / axis output slot).
 * @param[in]  out_dt Canonical dtype string of @p out.
 */
static void
cpu_reduce_sum_into(NDArray *src, NDArray *out, const char *out_dt) {
    const char *src_dt = NDArray_TYPE(src);
    long n = NDArray_NUMELEMENTS(src);
    if (!strcmp(out_dt, "int64")) {
        int64_t acc = 0;
        for (long i = 0; i < n; i++) {
            acc += ndarray_element_to_int64(src_dt, NDArray_DATA(src), (size_t)i);
        }
        memcpy(NDArray_DATA(out), &acc, sizeof(int64_t));
        return;
    }
    if (!strcmp(out_dt, "uint64")) {
        uint64_t acc = 0;
        for (long i = 0; i < n; i++) {
            acc += ndarray_element_to_uint64(src_dt, NDArray_DATA(src), (size_t)i);
        }
        memcpy(NDArray_DATA(out), &acc, sizeof(uint64_t));
        return;
    }
    if (!strcmp(out_dt, "float128")) {
        ndarray_fp128_t acc = NDARRAY_FP128_ZERO();
        for (long i = 0; i < n; i++) {
            ndarray_fp128_t v = ndarray_element_to_fp128(src_dt, NDArray_DATA(src), (size_t)i);
            acc = NDARRAY_FP128_ADD(acc, v);
        }
        memcpy(NDArray_DATA(out), &acc, NDARRAY_FP128_SIZE);
        return;
    }
    /* Float dtypes (f4/f8/f16/f32/f64): double accumulator rounds to the
       output's precision at the very end. */
    double acc = 0.0;
    for (long i = 0; i < n; i++) {
        acc += ndarray_element_to_double(src_dt, NDArray_DATA(src), (size_t)i);
    }
    ndarray_set_from_double(out_dt, NDArray_DATA(out), 0, acc);
}

/**
 * @brief CPU prod reduction — see `cpu_reduce_sum_into` for the accumulator
 *        selection rules. The combining op is `*` instead of `+`, identity is
 *        `1` instead of `0`, and float128 identity is `NDARRAY_FP128_FROM_D(1.0)`.
 *
 * @param[in]  src    Source NDArray.
 * @param[out] out    Destination NDArray.
 * @param[in]  out_dt Canonical dtype string of @p out.
 */
static void
cpu_reduce_prod_into(NDArray *src, NDArray *out, const char *out_dt) {
    const char *src_dt = NDArray_TYPE(src);
    long n = NDArray_NUMELEMENTS(src);
    if (!strcmp(out_dt, "int64")) {
        int64_t acc = 1;
        for (long i = 0; i < n; i++) {
            acc *= ndarray_element_to_int64(src_dt, NDArray_DATA(src), (size_t)i);
        }
        memcpy(NDArray_DATA(out), &acc, sizeof(int64_t));
        return;
    }
    if (!strcmp(out_dt, "uint64")) {
        uint64_t acc = 1;
        for (long i = 0; i < n; i++) {
            acc *= ndarray_element_to_uint64(src_dt, NDArray_DATA(src), (size_t)i);
        }
        memcpy(NDArray_DATA(out), &acc, sizeof(uint64_t));
        return;
    }
    if (!strcmp(out_dt, "float128")) {
        ndarray_fp128_t acc = NDARRAY_FP128_FROM_D(1.0);
        for (long i = 0; i < n; i++) {
            ndarray_fp128_t v = ndarray_element_to_fp128(src_dt, NDArray_DATA(src), (size_t)i);
            acc = NDARRAY_FP128_MUL(acc, v);
        }
        memcpy(NDArray_DATA(out), &acc, NDARRAY_FP128_SIZE);
        return;
    }
    double acc = 1.0;
    for (long i = 0; i < n; i++) {
        acc *= ndarray_element_to_double(src_dt, NDArray_DATA(src), (size_t)i);
    }
    ndarray_set_from_double(out_dt, NDArray_DATA(out), 0, acc);
}

/**
 * @brief Allocate a 0-D NDArray containing the no-axis reduction of @p a in
 *        @p a's widened reduction dtype, on the source device.
 *
 * The output dtype follows `ndarray_reduce_result_dtype` (NumPy default
 * widening — narrow ints widen to `int64`/`uint64`; floats keep their
 * dtype). The accumulator dtype on CPU matches the output for `int64` /
 * `uint64` / `float128` (native-precision accumulation) and is `double`
 * for the floating-point dtypes (matches NumPy's "sum precision = source
 * dtype").
 *
 * For GPU sources the work stays on device: the routine calls
 * `NDArray_Reduce_Sum`/`Prod` (which dispatches to the per-dtype
 * `cuda_reduce_*` kernels — only a single double crosses the bus) and
 * encodes the resulting double into the host-side widened slot. **Known
 * limitation**: when the GPU sum / prod overflows the double mantissa
 * (`> 2⁵³`) the result rounds; CPU and GPU may differ for `int64` /
 * `uint64` / `float128` inputs in that regime.
 *
 * @param[in] a  Source NDArray (any dtype, any device).
 * @param[in] op `ND_AXIS_RED_SUM` or `ND_AXIS_RED_PROD`.
 * @return Freshly-allocated 0-D NDArray on success, NULL on failure (with a
 *         PHP exception in flight).
 */
static NDArray *
ndarray_reduce_full_as_ndarray(NDArray *a, enum ndarray_reduce_axis_op op) {
    const char *src_dt = NDArray_TYPE(a);
    const char *out_dt = ndarray_reduce_result_dtype(src_dt);
    int *shape0 = emalloc(sizeof(int));
    shape0[0] = 1;
    NDArray *out = NDArray_Empty(shape0, 0, out_dt, NDARRAY_DEVICE_CPU);
    if (out == NULL) return NULL;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        if (op == ND_AXIS_RED_SUM)  cpu_reduce_sum_into (a, out, out_dt);
        else                         cpu_reduce_prod_into(a, out, out_dt);
        return out;
    }

    /* GPU source: run the typed `cuda_reduce_*` kernel that returns a single
       double, then encode into the widened CPU slot. The CPU/GPU mismatch
       for wide-int64 / wide-uint64 / wide-fp128 reductions is documented
       above. */
    double v = (op == ND_AXIS_RED_SUM) ? NDArray_Reduce_Sum(a)
                                       : NDArray_Reduce_Prod(a);
    if (EG(exception)) {
        NDArray_FREE(out);
        return NULL;
    }
    if (!strcmp(out_dt, "int64")) {
        int64_t iv = (int64_t)v;
        memcpy(NDArray_DATA(out), &iv, sizeof(int64_t));
    } else if (!strcmp(out_dt, "uint64")) {
        uint64_t uv = (uint64_t)v;
        memcpy(NDArray_DATA(out), &uv, sizeof(uint64_t));
    } else {
        ndarray_set_from_double(out_dt, NDArray_DATA(out), 0, v);
    }
    return out;
}

NDArray *NDArray_Reduce_Sum_AsNDArray(NDArray *a) {
    return ndarray_reduce_full_as_ndarray(a, ND_AXIS_RED_SUM);
}

NDArray *NDArray_Reduce_Prod_AsNDArray(NDArray *a) {
    return ndarray_reduce_full_as_ndarray(a, ND_AXIS_RED_PROD);
}

#ifdef HAVE_CUBLAS
/**
 * @brief Run @p a [i] op= @p b [i] for every element in place, on GPU.
 *
 * Dispatches to the dtype-appropriate `cuda_<op>_<tag>` kernel; both buffers
 * must already reside on GPU and be of the same dtype. The kernel writes
 * back into @p a (in-place), so callers driving the axis-reduction loop
 * never need a scratch buffer.
 *
 * `float128` is handled via the dd kernels (one fp128 element = 2 doubles
 * in the dd layout). `float4` / `float8` are not handled here — the caller
 * arranges for fp16 staging before driving this loop.
 *
 * @param[in]     dt Canonical dtype string of both buffers.
 * @param[in,out] a  GPU buffer (also the result).
 * @param[in]     b  GPU buffer.
 * @param[in]     n  Element count (logical: fp128 counts as one per slot).
 * @param[in]     op `ND_AXIS_RED_SUM` or `ND_AXIS_RED_PROD`.
 * @return 0 on success, -1 when the dtype has no native kernel.
 */
static int
gpu_axis_inplace_op(const char *dt, void *a, const void *b, int n,
                    enum ndarray_reduce_axis_op op) {
#define HANDLE_TYPED(DTSTR, TAG, T)                                                  \
    if (!strcmp(dt, DTSTR)) {                                                         \
        T *ap = (T *)a;                                                               \
        T *bp = (T *)b;                                                               \
        if (op == ND_AXIS_RED_SUM) cuda_add_##TAG(ap, bp, ap, n);                     \
        else                       cuda_mul_##TAG(ap, bp, ap, n);                     \
        return 0;                                                                     \
    }
    HANDLE_TYPED("int8",    i8,  int8_t)
    HANDLE_TYPED("uint8",   u8,  uint8_t)
    HANDLE_TYPED("int16",   i16, int16_t)
    HANDLE_TYPED("uint16",  u16, uint16_t)
    HANDLE_TYPED("int32",   i32, int32_t)
    HANDLE_TYPED("uint32",  u32, uint32_t)
    HANDLE_TYPED("int64",   i64, int64_t)
    HANDLE_TYPED("uint64",  u64, uint64_t)
    HANDLE_TYPED("float16", f16, uint16_t)
    HANDLE_TYPED("float64", f64, double)
#undef HANDLE_TYPED
    if (!strcmp(dt, "float32")) {
        float *ap = (float *)a;
        float *bp = (float *)b;
        if (op == ND_AXIS_RED_SUM) cuda_add_float(n, ap, bp, ap, n);
        else                       cuda_multiply_float(n, ap, bp, ap, n);
        return 0;
    }
    if (!strcmp(dt, "float128")) {
        double *ap = (double *)a;
        double *bp = (double *)b;
        if (op == ND_AXIS_RED_SUM) cuda_add_dd(ap, bp, ap, n);
        else                       cuda_mul_dd(ap, bp, ap, n);
        return 0;
    }
    return -1;
}
#endif /* HAVE_CUBLAS */

/**
 * @brief Reduce @p a along a single axis, preserving dtype and device.
 *
 * Implements numpy-style `sum(axis=k)` / `prod(axis=k)`. The output shape
 * is @p a's shape with axis @p axis removed; a 1-D input collapses to a
 * 0-D NDArray (callers route 0-D through `NDArray_ScalarToZval` to land in
 * a dtype-correct PHP scalar).
 *
 * Algorithm:
 *  - Resolve negative axes (numpy semantics: `-1` is the last axis).
 *  - Move @p axis to position 0 via `NDArray_Rollaxis` (no-op when already
 *    there); after the roll the per-slice memory layout is contiguous, so
 *    slice `k` lives at `data + k * n_per_slice * elsize`.
 *  - **CPU path**: walk each output position once, summing / multiplying
 *    `a_rolled[i + k * n_per_slice]` for `k` in `[0, s_axis)` with `double`
 *    accumulators (matches the no-axis CPU reduction's precision profile).
 *  - **GPU path**: initialise the output with `cudaMemcpy(D2D)` of slice 0,
 *    then apply the dtype's typed binop kernel in place for slices
 *    `1 .. s_axis - 1`. `float4` / `float8` are routed through a float16
 *    staging copy because no native fp4/fp8 binop kernels exist.
 *
 * Empty axis (`s_axis == 0`): return an output filled with the operation's
 * identity (zero for sum, one for prod) — same semantic as `np.sum(axis=0)`
 * on a zero-size dimension.
 *
 * @param[in] a    Source NDArray; any dtype, CPU or GPU.
 * @param[in] axis Axis index in `[-ndim, ndim)`.
 * @param[in] op   `ND_AXIS_RED_SUM` or `ND_AXIS_RED_PROD`.
 * @return Freshly-allocated NDArray on success, NULL on validation error or
 *         allocation failure (with a PHP exception in flight).
 */
NDArray *
NDArray_Reduce_Axis(NDArray *a, int axis, enum ndarray_reduce_axis_op op) {
    int ndim = NDArray_NDIM(a);
    if (ndim == 0) {
        /* 0-D source: numpy throws for any axis. */
        zend_throw_error(NULL, "axis %d is out of bounds for array of dimension 0", axis);
        return NULL;
    }
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        zend_throw_error(NULL,
                         "axis %d is out of bounds for array of dimension %d",
                         axis, ndim);
        return NULL;
    }

    const char *src_dt = NDArray_TYPE(a);
    const char *out_dt = ndarray_reduce_result_dtype(src_dt);
    int dev = NDArray_DEVICE(a);
    long s_axis = (long)NDArray_SHAPE(a)[axis];
    long total = (long)NDArray_NUMELEMENTS(a);
    long n_per_slice = (s_axis > 0) ? (total / s_axis) : 0;

    /* Output shape — input shape with `axis` removed. */
    int out_ndim = ndim - 1;
    int *out_shape = emalloc(sizeof(int) * (out_ndim > 0 ? out_ndim : 1));
    if (out_ndim > 0) {
        for (int i = 0, j = 0; i < ndim; i++) {
            if (i != axis) out_shape[j++] = NDArray_SHAPE(a)[i];
        }
    } else {
        out_shape[0] = 1;  /* placeholder; ndim == 0 ignores it */
    }

    /* Empty reduction axis (s_axis == 0): output is identity-filled. */
    if (s_axis == 0) {
        long out_n = 1;
        for (int i = 0; i < out_ndim; i++) out_n *= out_shape[i];
        NDArray *out = NDArray_Empty(out_shape, out_ndim, out_dt, dev);
        if (out == NULL) return NULL;
        double id_v = (op == ND_AXIS_RED_SUM) ? 0.0 : 1.0;
        if (dev == NDARRAY_DEVICE_CPU) {
            for (long i = 0; i < out_n; i++) {
                ndarray_set_from_double(out_dt, NDArray_DATA(out), (size_t)i, id_v);
            }
        }
#ifdef HAVE_CUBLAS
        else {
            int elsize = NDArray_ELSIZE(out);
            char *staging = ecalloc((size_t)(out_n > 0 ? out_n : 1), (size_t)elsize);
            for (long i = 0; i < out_n; i++) {
                ndarray_set_from_double(out_dt, staging, (size_t)i, id_v);
            }
            NDArray_TypedH2D((char *)NDArray_DATA(out), staging, out_n, out_dt);
            efree(staging);
        }
#endif
        return out;
    }

    /* Allocate output on the source device with the widened dtype. */
    NDArray *out = NDArray_Empty(out_shape, out_ndim, out_dt, dev);
    if (out == NULL) {
        return NULL;
    }

    /* CPU per-output-element walk: pick the accumulator dtype based on
       out_dt so wide-int / wide-fp128 reductions stay exact. */
    if (dev == NDARRAY_DEVICE_CPU) {
        /* Move axis to position 0 so each axis-0 slice is contiguous in
           memory — simplifies the per-output index math. */
        NDArray *a_rolled = (axis == 0) ? a : NDArray_Rollaxis(a, axis, 0);
        if (a_rolled == NULL) {
            NDArray_FREE(out);
            return NULL;
        }
        const char *r_dt = NDArray_TYPE(a_rolled);
        for (long i = 0; i < n_per_slice; i++) {
            if (!strcmp(out_dt, "int64")) {
                int64_t acc = (op == ND_AXIS_RED_SUM) ? 0 : 1;
                for (long k = 0; k < s_axis; k++) {
                    int64_t v = ndarray_element_to_int64(
                        r_dt, NDArray_DATA(a_rolled), (size_t)(i + k * n_per_slice));
                    if (op == ND_AXIS_RED_SUM) acc += v; else acc *= v;
                }
                memcpy((int64_t *)NDArray_DATA(out) + i, &acc, sizeof(int64_t));
            } else if (!strcmp(out_dt, "uint64")) {
                uint64_t acc = (op == ND_AXIS_RED_SUM) ? 0 : 1;
                for (long k = 0; k < s_axis; k++) {
                    uint64_t v = ndarray_element_to_uint64(
                        r_dt, NDArray_DATA(a_rolled), (size_t)(i + k * n_per_slice));
                    if (op == ND_AXIS_RED_SUM) acc += v; else acc *= v;
                }
                memcpy((uint64_t *)NDArray_DATA(out) + i, &acc, sizeof(uint64_t));
            } else if (!strcmp(out_dt, "float128")) {
                ndarray_fp128_t acc = (op == ND_AXIS_RED_SUM)
                    ? NDARRAY_FP128_ZERO()
                    : NDARRAY_FP128_FROM_D(1.0);
                for (long k = 0; k < s_axis; k++) {
                    ndarray_fp128_t v = ndarray_element_to_fp128(
                        r_dt, NDArray_DATA(a_rolled), (size_t)(i + k * n_per_slice));
                    acc = (op == ND_AXIS_RED_SUM) ? NDARRAY_FP128_ADD(acc, v)
                                                  : NDARRAY_FP128_MUL(acc, v);
                }
                memcpy((ndarray_fp128_t *)NDArray_DATA(out) + i, &acc, NDARRAY_FP128_SIZE);
            } else {
                /* Float dtypes: double accumulator stored via set_from_double. */
                double acc = (op == ND_AXIS_RED_SUM) ? 0.0 : 1.0;
                for (long k = 0; k < s_axis; k++) {
                    double v = ndarray_element_to_double(
                        r_dt, NDArray_DATA(a_rolled), (size_t)(i + k * n_per_slice));
                    if (op == ND_AXIS_RED_SUM) acc += v; else acc *= v;
                }
                ndarray_set_from_double(out_dt, NDArray_DATA(out), (size_t)i, acc);
            }
        }
        if (a_rolled != a) NDArray_FREE(a_rolled);
        return out;
    }
#ifdef HAVE_CUBLAS
    /* GPU path. Cast input to the widened dtype first if necessary so the
       accumulation runs entirely in the widened native int / fp dtype on
       device — narrow ints won't wrap at their narrow range and float128
       inputs accumulate with dd precision. */
    NDArray *a_widened = (strcmp(src_dt, out_dt) == 0)
        ? a : NDArray_AsType(a, out_dt);
    if (a_widened == NULL) {
        NDArray_FREE(out);
        return NULL;
    }
    NDArray *a_rolled = (axis == 0) ? a_widened
                                     : NDArray_Rollaxis(a_widened, axis, 0);
    if (a_rolled == NULL) {
        if (a_widened != a) NDArray_FREE(a_widened);
        NDArray_FREE(out);
        return NULL;
    }

    int elsize           = NDArray_ELSIZE(a_rolled);
    const char *src_data = (const char *)NDArray_DATA(a_rolled);
    char *out_data       = (char *)NDArray_DATA(out);
    int n_slice_i        = (int)n_per_slice;
    size_t slice_bytes   = (size_t)n_per_slice * (size_t)elsize;

    /* `out_dt` is the widened dtype produced by `ndarray_reduce_result_dtype`
       — narrow floats (fp4 / fp8 / fp16) have already been widened to
       float32 by the `NDArray_AsType(a, out_dt)` cast above, so the GPU
       loop below only ever sees the native-kernel dtypes (int8..int64,
       uint8..uint64, float32, float64, float128 via dd). No fp4 / fp8
       branch is needed here. */
    cudaError_t cerr = cudaMemcpy(out_data, src_data, slice_bytes,
                                   cudaMemcpyDeviceToDevice);
    if (cerr != cudaSuccess) {
        if (a_rolled != a_widened) NDArray_FREE(a_rolled);
        if (a_widened != a) NDArray_FREE(a_widened);
        NDArray_FREE(out);
        zend_throw_error(NULL,
            "GPU axis reduction memcpy failed: %s", cudaGetErrorString(cerr));
        return NULL;
    }
    for (long k = 1; k < s_axis; k++) {
        const void *src_k = (const void *)(src_data + (size_t)k * slice_bytes);
        if (gpu_axis_inplace_op(out_dt, out_data, src_k, n_slice_i, op) < 0) {
            if (a_rolled != a_widened) NDArray_FREE(a_rolled);
            if (a_widened != a) NDArray_FREE(a_widened);
            NDArray_FREE(out);
            zend_throw_error(NULL,
                "GPU axis reduction has no kernel for dtype \"%s\"", out_dt);
            return NULL;
        }
    }
    if (a_rolled != a_widened) NDArray_FREE(a_rolled);
    if (a_widened != a) NDArray_FREE(a_widened);
    return out;
#else
    NDArray_FREE(out);
    zend_throw_error(NULL, "GPU reduction unavailable: built without CUDA");
    return NULL;
#endif
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

#if HAVE_AVX2
__m256 fix_negative_zero(__m256 vec) {
    __m256 zero = _mm256_set1_ps(-0.0f);
    __m256 mask = _mm256_cmp_ps(vec, zero, _CMP_EQ_OQ);
    return _mm256_blendv_ps(vec, zero, mask);
}
#endif

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

/* EXPRs go through the NDARRAY_FP128_* macros from ndarray_types.h so they
   resolve to native __float128 operators on the libquadmath build and to
   ndarray_dd_* calls on the DD-emulated build. Same source code, same
   precision tier per platform. */
DEFINE_FP128_BINOP(Add,       NDARRAY_FP128_ADD(x, y))
DEFINE_FP128_BINOP(Subtract,  NDARRAY_FP128_SUB(x, y))
DEFINE_FP128_BINOP(Multiply,  NDARRAY_FP128_MUL(x, y))
DEFINE_FP128_BINOP(Divide,    NDARRAY_FP128_DIV(x, y))

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
#elif NDARRAY_HAVE_FLOAT128
        rd[i] = (ndarray_fp128_t)powl((long double)ad[i], (long double)bd[i]);
#else
        /* DD path: integer exponents stay at full DD precision; fractional
           exponents degrade to fp64 (DD log+exp would add ~400 lines and
           is unused by current tests). */
        rd[i] = ndarray_dd_pow(ad[i], bd[i]);
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
#elif NDARRAY_HAVE_FLOAT128
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
#else
        /* DD path: same fmod identity, implemented via dd_math.h primitives
           (trunc handles the int64-fits-in-mantissa edge case too). */
        rd[i] = ndarray_dd_fmod(ad[i], bd[i]);
#endif
    }
    if (a_temp) NDArray_FREE(a);
    if (b_temp) NDArray_FREE(b);
    if (broadcasted) NDArray_FREE(broadcasted);
    return result;
}

/* ── Two-argument arctangent (atan2) CPU kernels ──────────────────────────
   atan2 is a binary element-wise op that always returns a floating-point
   result, so `ndarray_promote_and_op` promotes both operands to a common
   float compute dtype (float32 / float64 / float128) before dispatching
   here — these kernels never see integer input. Broadcasting and the 0-D
   scalar case reuse `NDArray_Broadcast`, matching the existing float
   arithmetic kernels (`NDArray_Add_Double` et al.) so CPU and the
   `cuda_atan2_*` GPU kernels stay in step on every shape they both support.
   CPU-only: GPU residency is handled by `NDArray_TypedBinOp_GPU`.

   Argument order follows NumPy: `arctan2(a, b)` == C `atan2(a, b)` (a is the
   numerator / y-coordinate, b the denominator / x-coordinate). */
#define DEFINE_ATAN2_FLOAT_CPU(NAME, T, DT_CONST, FN)                              \
NDArray* NAME(NDArray* a, NDArray* b) {                                            \
    /* Expand a 0-D scalar operand to the peer's shape first, matching            \
       NDArray_Add_Double et al. AND the GPU path: NumPy/PyTorch broadcast        \
       arctan2(0-D, shape-(n,)) to (n,), never to a 0-D scalar. Without this,     \
       a 0-D vs numel-1-array pair would tie on element count, fall through to    \
       the `else` below, and take the 0-D operand's rank — yielding a 0-D CPU     \
       result while the GPU (which broadcasts the scalar) returns (n,).           \
       NDArray_Broadcast replicates the single element for any dtype. */          \
    NDArray *sa = NULL, *sb = NULL;                                               \
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {                            \
        sa = NDArray_Broadcast(a, b); if (sa == NULL) return NULL; a = sa;        \
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {                     \
        sb = NDArray_Broadcast(b, a); if (sb == NULL) return NULL; b = sb;        \
    }                                                                             \
    NDArray *broadcasted = NULL, *a_broad, *b_broad;                              \
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {                        \
        broadcasted = NDArray_Broadcast(a, b);                                    \
        if (broadcasted == NULL) {                                                \
            if (sa) NDArray_FREE(sa);                                             \
            if (sb) NDArray_FREE(sb);                                             \
            return NULL;                                                          \
        }                                                                         \
        a_broad = broadcasted; b_broad = b;                                       \
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {                 \
        broadcasted = NDArray_Broadcast(b, a);                                    \
        if (broadcasted == NULL) {                                                \
            if (sa) NDArray_FREE(sa);                                             \
            if (sb) NDArray_FREE(sb);                                             \
            return NULL;                                                          \
        }                                                                         \
        b_broad = broadcasted; a_broad = a;                                       \
    } else { a_broad = a; b_broad = b; }                                          \
    int ndim   = NDArray_NDIM(a_broad);                                           \
    int *shape = emalloc(sizeof(int) * (ndim > 0 ? ndim : 1));                    \
    if (ndim > 0) memcpy(shape, NDArray_SHAPE(a_broad), sizeof(int) * ndim);      \
    else          shape[0] = 1;                                                   \
    NDArray *result = NDArray_Empty(shape, ndim, DT_CONST, NDARRAY_DEVICE_CPU);   \
    if (result == NULL) {                                                         \
        if (broadcasted) NDArray_FREE(broadcasted);                               \
        if (sa) NDArray_FREE(sa);                                                 \
        if (sb) NDArray_FREE(sb);                                                 \
        return NULL;                                                              \
    }                                                                             \
    T       *rd = (T *)NDArray_DATA(result);                                      \
    const T *ad = (const T *)NDArray_DATA(a_broad);                               \
    const T *bd = (const T *)NDArray_DATA(b_broad);                               \
    long n = NDArray_NUMELEMENTS(result);                                         \
    for (long i = 0; i < n; i++) rd[i] = FN(ad[i], bd[i]);                        \
    if (broadcasted) NDArray_FREE(broadcasted);                                   \
    if (sa) NDArray_FREE(sa);                                                     \
    if (sb) NDArray_FREE(sb);                                                     \
    return result;                                                                \
}
DEFINE_ATAN2_FLOAT_CPU(NDArray_Arctan2_Float,  float,  NDARRAY_TYPE_FLOAT32, atan2f)
DEFINE_ATAN2_FLOAT_CPU(NDArray_Arctan2_Double, double, NDARRAY_TYPE_FLOAT64, atan2)
#undef DEFINE_ATAN2_FLOAT_CPU

/* float128 atan2 — same broadcast / 0-D handling as the other fp128 binops
   via DEFINE_FP128_BINOP; the per-element body routes through
   `NDARRAY_FP128_ATAN2`, which is `atan2q` on the libquadmath build (full
   113-bit) and the DD `atan2(double)` fallback elsewhere. */
DEFINE_FP128_BINOP(Arctan2, NDARRAY_FP128_ATAN2(x, y))

/* ──────────────────────────────────────────────────────────────────────────
   Native integer CPU arithmetic kernels covering every one of the eight
   integer dtypes: `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`,
   `int64`, `uint64`.

   The legacy dispatch promoted every integer dtype to `float32` (narrow
   ints) or `float64` (int32 / uint32 / int64 / uint64) before computing,
   then cast the result back. That round-trip:
    1. Loses precision once the intermediate exceeds the float mantissa
       (24 bits for narrow ints — still safe for int8..int16 products —
       53 bits for int32 / uint32 / int64 / uint64; int32 * int32 with
       products past 2⁵³ silently rounds; int64 / uint64 past 2⁵³ rounds
       in every binary op).
    2. Diverges from PyTorch which always computes natively in the input
       dtype with C-style modular wrap.
    3. Diverges between CPU (double rounds, then narrow cast wraps) and
       GPU (`cuda_cast_f64_to_i32` saturates on out-of-range doubles —
       different result on the same inputs).

   The kernels below operate directly on the native integer storage so
   the result matches PyTorch's modular semantics on both CPU and GPU.
   Identical broadcast / 0-D scalar / shape handling as the float kernels.
   ────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Broadcast a 0-D scalar to @p other's shape on CPU, preserving its
 *        dtype.
 *
 * Shared by the integer-binop path so the scalar+array case can fall into
 * the same per-element loop as the array+array case. The caller frees the
 * returned NDArray.
 *
 * @param[in] scalar 0-D NDArray (dtype must match @p other's).
 * @param[in] other  Peer NDArray whose shape drives the broadcast.
 * @return Caller-owned broadcast NDArray on success, NULL on allocation
 *         failure (PHP exception in flight).
 */
static NDArray *
ndarray_int_broadcast_scalar_cpu(NDArray *scalar, NDArray *other) {
    int ndim = NDArray_NDIM(other);
    int *n_shape = emalloc(sizeof(int) * (ndim > 0 ? ndim : 1));
    if (ndim > 0) {
        memcpy(n_shape, NDArray_SHAPE(other), sizeof(int) * ndim);
    } else {
        n_shape[0] = 1;
    }
    NDArray *r = NDArray_Empty(n_shape, ndim, NDArray_TYPE(other),
                                NDARRAY_DEVICE_CPU);
    if (r == NULL) return NULL;
    long n = NDArray_NUMELEMENTS(r);
    int elsize = NDArray_ELSIZE(other);
    const char *src = (const char *)NDArray_DATA(scalar);
    char *dst = (char *)NDArray_DATA(r);
    for (long i = 0; i < n; i++) {
        memcpy(dst + (size_t)i * (size_t)elsize, src, (size_t)elsize);
    }
    return r;
}

/**
 * @brief Compile a single per-dtype native-int binop body.
 *
 * The macro takes a C type (signed) and an unsigned-twin type for the
 * arithmetic ops that need defined wrap-around semantics (signed integer
 * overflow is undefined behaviour in C, so add/sub/mul/pow go through the
 * unsigned twin and round-trip back; div/mod use the signed type so the
 * sign-of-dividend remainder is what falls out).
 *
 * @param T     Signed integer C type (e.g. `int32_t`).
 * @param UT    Unsigned-twin C type of the same width (e.g. `uint32_t`).
 *
 * For unsigned target dtypes (`uint8` etc.) we pass the unsigned type as
 * both T and UT — the casts then collapse to no-ops, and `%` on unsigned
 * is already well-defined.
 *
 * The generated body assumes `n` elements, `ap` / `bp` source pointers
 * of type `T *`, and `rp` destination pointer of type `T *`.
 */
#define NDARRAY_INT_BINOP_BODY(T, UT)                                                \
    do {                                                                              \
        switch (opcode) {                                                             \
            case ZEND_ADD:                                                            \
                for (long i = 0; i < n; i++) {                                        \
                    rp[i] = (T)((UT)ap[i] + (UT)bp[i]);                               \
                }                                                                     \
                break;                                                                \
            case ZEND_SUB:                                                            \
                for (long i = 0; i < n; i++) {                                        \
                    rp[i] = (T)((UT)ap[i] - (UT)bp[i]);                               \
                }                                                                     \
                break;                                                                \
            case ZEND_MUL:                                                            \
                for (long i = 0; i < n; i++) {                                        \
                    rp[i] = (T)((UT)ap[i] * (UT)bp[i]);                               \
                }                                                                     \
                break;                                                                \
            case ZEND_MOD:                                                            \
                /* Divisor==0 → 0 to avoid SIGFPE; matches the float */               \
                /* kernels' NaN convention. C11 `%` on signed values is */            \
                /* truncated (sign of dividend), matching PyTorch's */                \
                /* `torch.remainder` semantics for integer dtypes. */                 \
                for (long i = 0; i < n; i++) {                                        \
                    rp[i] = (bp[i] == 0) ? 0 : (T)(ap[i] % bp[i]);                    \
                }                                                                     \
                break;                                                                \
            case ZEND_POW: {                                                          \
                /* Binary exponentiation in unsigned width — base / r */              \
                /* round-trip through UT so signed overflow stays UB-free. */         \
                /* Negative exponents on signed types yield 0 (integer */             \
                /* truncation of a fractional reciprocal), matching */                \
                /* PyTorch's int-pow contract. */                                     \
                for (long i = 0; i < n; i++) {                                        \
                    T base = ap[i], exp = bp[i];                                      \
                    if (exp < 0) { rp[i] = 0; continue; }                             \
                    UT r = 1u;                                                        \
                    UT b_acc = (UT)base;                                              \
                    while (exp > 0) {                                                 \
                        if (exp & 1) r = (UT)(r * b_acc);                             \
                        b_acc = (UT)(b_acc * b_acc);                                  \
                        exp >>= 1;                                                    \
                    }                                                                 \
                    rp[i] = (T)r;                                                     \
                }                                                                     \
                break;                                                                \
            }                                                                         \
            default:                                                                  \
                zend_throw_error(NULL,                                                \
                    "Unsupported opcode for native int CPU binop.");                  \
                return -1;                                                            \
        }                                                                             \
    } while (0)

/**
 * @brief Run the per-dtype native-int binop body for @p dt against the
 *        already-broadcast @p ap / @p bp / @p rp buffers.
 *
 * Centralises the dtype dispatch so the broadcast / alloc plumbing in
 * `ndarray_int_binop_cpu` doesn't have to repeat the 8-way switch.
 *
 * @param[in]  dt     Canonical dtype string (one of `int8`..`uint64`).
 * @param[in]  n      Element count.
 * @param[in]  ap, bp Operand pointers; must be of `dt`'s native width.
 * @param[out] rp     Destination pointer; same width as @p ap / @p bp.
 * @param[in]  opcode ZEND_ADD / SUB / MUL / MOD / POW.
 * @return 0 on success, -1 on dispatch error (PHP exception in flight).
 */
static int
ndarray_run_int_binop_typed(const char *dt, long n,
                             const void *ap_v, const void *bp_v, void *rp_v,
                             int opcode) {
#define DISPATCH_INT_TYPED(STR, T, UT)                                                \
    if (!strcmp(dt, STR)) {                                                           \
        const T *ap = (const T *)ap_v;                                                \
        const T *bp = (const T *)bp_v;                                                \
        T       *rp = (T       *)rp_v;                                                \
        NDARRAY_INT_BINOP_BODY(T, UT);                                                \
        return 0;                                                                     \
    }
    DISPATCH_INT_TYPED("int8",   int8_t,   uint8_t)
    DISPATCH_INT_TYPED("uint8",  uint8_t,  uint8_t)
    DISPATCH_INT_TYPED("int16",  int16_t,  uint16_t)
    DISPATCH_INT_TYPED("uint16", uint16_t, uint16_t)
    DISPATCH_INT_TYPED("int32",  int32_t,  uint32_t)
    DISPATCH_INT_TYPED("uint32", uint32_t, uint32_t)
    DISPATCH_INT_TYPED("int64",  int64_t,  uint64_t)
    DISPATCH_INT_TYPED("uint64", uint64_t, uint64_t)
#undef DISPATCH_INT_TYPED
    zend_throw_error(NULL,
        "Native CPU int binop: unsupported dtype \"%s\".", dt);
    return -1;
}

/**
 * @brief Broadcast / alloc / dispatch plumbing for native int CPU binops.
 *
 * Handles the 0-D scalar promotion, NumPy-style shape broadcasting,
 * result allocation, and per-dtype kernel dispatch shared by every
 * integer dtype. Inputs must already share the same dtype — the caller
 * (`ndarray_promote_and_op` via `NDArray_TypedBinOp_CPU_Int`) is
 * responsible for casting.
 *
 * @param[in] a, b    Same-dtype CPU operands; one may be 0-D.
 * @param[in] opcode  ZEND_ADD / SUB / MUL / MOD / POW.
 * @return Result NDArray on success (caller owns), NULL on validation /
 *         allocation failure (PHP exception in flight).
 */
static NDArray *
ndarray_int_binop_cpu(NDArray *a, NDArray *b, int opcode) {
    NDArray *a_temp = NULL, *b_temp = NULL, *broadcasted = NULL;
    const char *out_dt = NDArray_TYPE(a);

    /* 0-D scalar broadcast — fill a buffer of the peer's shape with the
       scalar's single element. */
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        a = ndarray_int_broadcast_scalar_cpu(a_temp, b);
        if (a == NULL) return NULL;
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        b = ndarray_int_broadcast_scalar_cpu(b_temp, a);
        if (b == NULL) return NULL;
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
        a_broad = a;
        b_broad = b;
    }
    if (a_broad == NULL || b_broad == NULL) {
        if (a_temp) NDArray_FREE(a);
        if (b_temp) NDArray_FREE(b);
        if (broadcasted) NDArray_FREE(broadcasted);
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    /* Allocate result mirroring a_broad's shape and dtype. */
    int *res_shape = emalloc(sizeof(int) * (NDArray_NDIM(a_broad) > 0
                                              ? NDArray_NDIM(a_broad) : 1));
    if (NDArray_NDIM(a_broad) > 0) {
        memcpy(res_shape, NDArray_SHAPE(a_broad),
               sizeof(int) * NDArray_NDIM(a_broad));
    } else {
        res_shape[0] = 1;
    }
    NDArray *result = NDArray_Empty(res_shape, NDArray_NDIM(a_broad),
                                     out_dt, NDARRAY_DEVICE_CPU);
    if (result == NULL) {
        if (a_temp) NDArray_FREE(a);
        if (b_temp) NDArray_FREE(b);
        if (broadcasted) NDArray_FREE(broadcasted);
        return NULL;
    }

    long n = NDArray_NUMELEMENTS(a_broad);
    if (ndarray_run_int_binop_typed(out_dt, n,
                                     NDArray_DATA(a_broad),
                                     NDArray_DATA(b_broad),
                                     NDArray_DATA(result),
                                     opcode) < 0) {
        NDArray_FREE(result);
        if (a_temp) NDArray_FREE(a);
        if (b_temp) NDArray_FREE(b);
        if (broadcasted) NDArray_FREE(broadcasted);
        return NULL;
    }

    if (a_temp) NDArray_FREE(a);
    if (b_temp) NDArray_FREE(b);
    if (broadcasted) NDArray_FREE(broadcasted);
    return result;
}

/**
 * @brief CPU binary-op dispatcher for every native integer dtype.
 *
 * Routes the supported opcodes to the native-int kernel above so:
 *  - PyTorch's modular wrap-around semantics are honoured exactly,
 *  - precision past 2⁵³ survives for `int64` / `uint64`,
 *  - the result matches the GPU path's native-int kernels bit-for-bit.
 *
 * The previous implementation handled only `int64` / `uint64`; narrower
 * ints still went through float promotion, which silently diverged from
 * PyTorch for `int32 * int32` once the intermediate exceeded the float64
 * mantissa.
 *
 * Both operands must already share the same dtype — the calling
 * dispatcher (`ndarray_promote_and_op`) handles the cast through
 * `NDArray_AsType` before reaching this entry point. Falls back to
 * NULL with a PHP error for opcodes outside the supported set; `/` (and
 * `arctan2`) are already promoted to a float dtype by
 * `ndarray_widen_int_to_float` and never reach here.
 *
 * @param[in] opcode ZEND_ADD / SUB / MUL / MOD / POW.
 * @param[in] a, b   Same-dtype operands on CPU; one may be 0-D.
 * @return Result NDArray on success, NULL on error.
 */
NDArray *
NDArray_TypedBinOp_CPU_Int(int opcode, NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_CPU
        || NDArray_DEVICE(b) != NDARRAY_DEVICE_CPU) {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_CPU_Int: both operands must be on CPU.");
        return NULL;
    }
    if (strcmp(NDArray_TYPE(a), NDArray_TYPE(b)) != 0) {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_CPU_Int: dtype mismatch (%s vs %s).",
            NDArray_TYPE(a), NDArray_TYPE(b));
        return NULL;
    }
    const char *dt = NDArray_TYPE(a);
    int is_int = (!strcmp(dt, "int8")  || !strcmp(dt, "uint8")  ||
                  !strcmp(dt, "int16") || !strcmp(dt, "uint16") ||
                  !strcmp(dt, "int32") || !strcmp(dt, "uint32") ||
                  !strcmp(dt, "int64") || !strcmp(dt, "uint64"));
    if (!is_int) {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_CPU_Int: unsupported dtype \"%s\".", dt);
        return NULL;
    }
    return ndarray_int_binop_cpu(a, b, opcode);
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

    /* For dd128, scalar_cpu's CPU storage may be __float128 (Linux GCC) or
       a (hi, lo) double-double pair (every other platform). Split into the
       (hi, lo) doubles the GPU kernel expects either way. */
    if (!strcmp(dt, "float128")) {
        ndarray_fp128_t v;
        memcpy(&v, scalar_cpu->data, NDARRAY_FP128_SIZE);
#if NDARRAY_HAVE_FLOAT128
        double hi = (double)v;
        double lo = (double)(v - (ndarray_fp128_t)hi);
#else
        double hi = v.hi;
        double lo = v.lo;
#endif
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

    /* atan2 always computes in a float dtype (the dispatcher promotes integer
       and narrow-float inputs to float32 / float64 / float128 first), so it
       dispatches on its own here: the shared DISPATCH_OP macro below is also
       instantiated for the integer dtypes, which have no `cuda_atan2_<int>`
       kernel. fp4 / fp8 never reach this point — their arithmetic compute
       dtype is float32. */
    if (opcode == NDARRAY_BINOP_ATAN2) {
        if (!strcmp(dt, "float32")) {
            cuda_atan2_f32((float *)NDArray_DATA(a_broad),
                           (float *)NDArray_DATA(b_broad),
                           (float *)NDArray_DATA(result), n);
        } else if (!strcmp(dt, "float64")) {
            cuda_atan2_f64((double *)NDArray_DATA(a_broad),
                           (double *)NDArray_DATA(b_broad),
                           (double *)NDArray_DATA(result), n);
        } else if (!strcmp(dt, "float128")) {
            cuda_atan2_dd((double *)NDArray_DATA(a_broad),
                          (double *)NDArray_DATA(b_broad),
                          (double *)NDArray_DATA(result), n);
        } else {
            NDArray_FREE(result);
            result = NULL;
            zend_throw_error(NULL,
                "GPU atan2: unsupported compute dtype %s.", dt);
        }
        goto done;
    }

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

/* ──────────────────────────────────────────────────────────────────────────
   Typed unary op dispatch — element-wise abs / negative / positive /
   reciprocal / sign / sqrt / rsqrt / square / clip / sinc across every
   supported dtype on both CPU and GPU.

   The dispatcher decides the result dtype per-op:
   - sqrt / rsqrt / reciprocal / sinc promote integer inputs to float64
     (32/64-bit ints) or float32 (narrow ints) — matches PyTorch's
     `result_type` for transcendental ops applied to integer tensors;
   - every other op preserves the input dtype.

   Narrow floats (`float4`, `float8`) route their compute through
   `float32` (no native intrinsics). The on-CPU path uses
   `NDArray_AsType` to stage; the on-GPU path uses the same AsType which
   stays on GPU for those dtypes.

   `clip_min` / `clip_max` are decimal strings parsed losslessly into the
   computed dtype so `float128` / `uint64` survive end-to-end.
   ────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Test whether @p dt names an integer dtype.
 * @param[in] dt Canonical dtype string.
 * @return 1 if @p dt is one of `int8..int64`/`uint8..uint64`; 0 otherwise.
 */
static int unary_is_int_dtype(const char *dt) {
    return (!strcmp(dt, "int8")   || !strcmp(dt, "uint8")  ||
            !strcmp(dt, "int16")  || !strcmp(dt, "uint16") ||
            !strcmp(dt, "int32")  || !strcmp(dt, "uint32") ||
            !strcmp(dt, "int64")  || !strcmp(dt, "uint64"));
}

/**
 * @brief Test whether @p dt is a 32/64-bit signed or unsigned integer dtype.
 * @param[in] dt Canonical dtype string.
 * @return 1 if @p dt is int32/uint32/int64/uint64; 0 otherwise.
 */
static int unary_is_wide_int_dtype(const char *dt) {
    return (!strcmp(dt, "int32") || !strcmp(dt, "uint32") ||
            !strcmp(dt, "int64") || !strcmp(dt, "uint64"));
}

/**
 * @brief Test whether @p op belongs to the trig / hyperbolic / angle /
 *        rounding family handled by `UNARY_TRIG_BODY`.
 *
 * The enum reserves a contiguous block (`SIN` .. `CEIL`) so the test
 * is a single range check. The float CPU dispatcher uses this to
 * pick between `UNARY_FLOAT_BODY` (basic + exp/log) and
 * `UNARY_TRIG_BODY` (trig/hyperbolic/angle/rounding), keeping each
 * macro focused.
 *
 * @param[in] op Unary op selector.
 * @return 1 if @p op is in the trig family, 0 otherwise.
 */
static int unary_op_is_trig(NDArrayUnaryOp op) {
    return op >= NDARRAY_UNOP_SIN && op <= NDARRAY_UNOP_CEIL;
}

/**
 * @brief Result dtype the unary op writes when applied to @p input_dt.
 *
 * sqrt / rsqrt / reciprocal / sinc / exp{,2,m1} / log{,1p,2,10,b} and
 * the trig/hyperbolic/angle family (sin/cos/tan/asin/acos/atan,
 * sinh/cosh/tanh/asinh/acosh/atanh, degrees, radians) promote integer
 * inputs to a floating-point dtype (PyTorch widening rule): narrow
 * ints (`int8`..`uint16`) widen to `float32`, the wider 32/64-bit
 * ints widen to `float64`. The rounding family (rint, fix, trunc,
 * floor, ceil) preserves integer dtypes — an integer is already its
 * own rounded value. Every other op preserves the input dtype.
 *
 * @param[in] op       Unary op selector.
 * @param[in] input_dt Source dtype string.
 * @return Canonical dtype string for the result NDArray.
 */
static const char *unary_result_dtype(NDArrayUnaryOp op, const char *input_dt) {
    switch (op) {
        case NDARRAY_UNOP_SQRT:
        case NDARRAY_UNOP_RSQRT:
        case NDARRAY_UNOP_RECIPROCAL:
        case NDARRAY_UNOP_SINC:
        case NDARRAY_UNOP_EXP:
        case NDARRAY_UNOP_EXP2:
        case NDARRAY_UNOP_EXPM1:
        case NDARRAY_UNOP_LOG:
        case NDARRAY_UNOP_LOG2:
        case NDARRAY_UNOP_LOG10:
        case NDARRAY_UNOP_LOG1P:
        case NDARRAY_UNOP_LOGB:
        case NDARRAY_UNOP_SIN:
        case NDARRAY_UNOP_COS:
        case NDARRAY_UNOP_TAN:
        case NDARRAY_UNOP_ARCSIN:
        case NDARRAY_UNOP_ARCCOS:
        case NDARRAY_UNOP_ARCTAN:
        case NDARRAY_UNOP_SINH:
        case NDARRAY_UNOP_COSH:
        case NDARRAY_UNOP_TANH:
        case NDARRAY_UNOP_ARCSINH:
        case NDARRAY_UNOP_ARCCOSH:
        case NDARRAY_UNOP_ARCTANH:
        case NDARRAY_UNOP_DEGREES:
        case NDARRAY_UNOP_RADIANS:
            if (unary_is_int_dtype(input_dt)) {
                return unary_is_wide_int_dtype(input_dt)
                    ? NDARRAY_TYPE_FLOAT64
                    : NDARRAY_TYPE_FLOAT32;
            }
            return input_dt;
        default:
            /* RINT / FIX / TRUNC / FLOOR / CEIL / ROUND preserve dtype —
               integers are already integer-valued, so the rounded result
               == input. */
            return input_dt;
    }
}

/**
 * @brief Choose the compute dtype that backs @p result_dt for unary ops.
 *
 * Narrow non-half floats (`float4`, `float8`) have no GPU intrinsics
 * and only minimal CPU range, so the dispatcher casts up to `float32`
 * for compute and casts back to the source dtype after the op. Every
 * other dtype computes natively.
 */
static const char *unary_compute_dtype(const char *result_dt) {
    if (!strcmp(result_dt, "float4") || !strcmp(result_dt, "float8")) {
        return NDARRAY_TYPE_FLOAT32;
    }
    return result_dt;
}

/* ── CPU per-dtype kernels (templated by macro) ─────────────────────────── */

/**
 * @brief CPU integer in-place unary loop.
 *
 * Wraps `signed` overflow through `unsigned` of the same width so:
 *  - `negate(INT_MIN)` returns `INT_MIN` (modular wrap, matches NumPy);
 *  - `abs(INT_MIN)`    returns `INT_MIN` (likewise);
 *  - `square` of any int wraps to the modular value PyTorch produces.
 */
#define UNARY_INT_BODY(T, UT, OP_TAG, LO_VAL, HI_VAL)                                \
    do {                                                                              \
        T *p = (T *)data;                                                             \
        for (long i = 0; i < n; i++) {                                                \
            T x = p[i];                                                               \
            switch (OP_TAG) {                                                         \
                case NDARRAY_UNOP_NEGATIVE:                                           \
                    p[i] = (T)(UT)(-(UT)x);                                           \
                    break;                                                            \
                case NDARRAY_UNOP_ABS:                                                \
                    p[i] = (x < (T)0) ? (T)(UT)(-(UT)x) : x;                          \
                    break;                                                            \
                case NDARRAY_UNOP_POSITIVE:                                           \
                    p[i] = x;                                                         \
                    break;                                                            \
                case NDARRAY_UNOP_SIGN:                                               \
                    p[i] = (T)((x > (T)0) - (x < (T)0));                              \
                    break;                                                            \
                case NDARRAY_UNOP_SQUARE:                                             \
                    p[i] = (T)(UT)((UT)x * (UT)x);                                    \
                    break;                                                            \
                case NDARRAY_UNOP_CLIP:                                               \
                    /* PyTorch clamp = min(max(x, lo), hi). For ints we use the   \
                       same branchless ordering as the float body so `lo > hi`   \
                       deterministically returns `hi` (matches PyTorch's docs). */\
                    {                                                                 \
                        T _y = (x < (LO_VAL)) ? (LO_VAL) : x;                        \
                        p[i] = ((HI_VAL) < _y) ? (HI_VAL) : _y;                      \
                    }                                                                 \
                    break;                                                            \
                default: break;                                                       \
            }                                                                         \
        }                                                                             \
    } while (0)

/**
 * @brief CPU floating-point in-place unary loop (float32 / float64 templated).
 *
 * Handles every op including the ones that require floating-point math
 * (`sqrt`, `rsqrt`, `reciprocal`, `sinc`, `exp`/`exp2`/`expm1`,
 * `log`/`log1p`/`log2`/`log10`/`logb`). Sign is implemented as the
 * branchless three-way comparison so it returns +0/-0/+1/-1 in the
 * source dtype.
 *
 * The libm helpers are passed in by suffix-aware overload (`expf` /
 * `exp` etc.) so the same macro instantiates against `float` and
 * `double` without an extra typename hop.
 */
#define UNARY_FLOAT_BODY(T, FN_SQRT, FN_SIN, FN_FABS,                                 \
                         FN_EXP, FN_EXP2, FN_EXPM1,                                   \
                         FN_LOG, FN_LOG1P, FN_LOG2, FN_LOG10, FN_LOGB,                \
                         OP_TAG, LO_VAL, HI_VAL)                                      \
    do {                                                                              \
        T *p = (T *)data;                                                             \
        for (long i = 0; i < n; i++) {                                                \
            T x = p[i];                                                               \
            switch (OP_TAG) {                                                         \
                case NDARRAY_UNOP_NEGATIVE:    p[i] = -x;                             \
                    break;                                                            \
                case NDARRAY_UNOP_ABS:         p[i] = FN_FABS(x);                     \
                    break;                                                            \
                case NDARRAY_UNOP_POSITIVE:    p[i] = x;                              \
                    break;                                                            \
                case NDARRAY_UNOP_SIGN:                                               \
                    /* PyTorch: NaN propagates through sign. The branchless     \
                       `(x > 0) - (x < 0)` idiom would return 0 (both ordered  \
                       comparisons are false for NaN), so we guard for NaN     \
                       explicitly. `x != x` is the canonical IEEE 754 NaN     \
                       test that works for every float dtype (fp16 / fp32 /    \
                       fp64 / fp128). */                                       \
                    if (x != x) p[i] = x;                                         \
                    else        p[i] = (T)((x > (T)0) - (x < (T)0));              \
                    break;                                                            \
                case NDARRAY_UNOP_SQUARE:      p[i] = x * x;                          \
                    break;                                                            \
                case NDARRAY_UNOP_RECIPROCAL:  p[i] = (T)1 / x;                       \
                    break;                                                            \
                case NDARRAY_UNOP_SQRT:        p[i] = FN_SQRT(x);                     \
                    break;                                                            \
                case NDARRAY_UNOP_RSQRT:       p[i] = (T)1 / FN_SQRT(x);              \
                    break;                                                            \
                case NDARRAY_UNOP_EXP:         p[i] = FN_EXP(x);                      \
                    break;                                                            \
                case NDARRAY_UNOP_EXP2:        p[i] = FN_EXP2(x);                     \
                    break;                                                            \
                case NDARRAY_UNOP_EXPM1:       p[i] = FN_EXPM1(x);                    \
                    break;                                                            \
                case NDARRAY_UNOP_LOG:         p[i] = FN_LOG(x);                      \
                    break;                                                            \
                case NDARRAY_UNOP_LOG1P:       p[i] = FN_LOG1P(x);                    \
                    break;                                                            \
                case NDARRAY_UNOP_LOG2:        p[i] = FN_LOG2(x);                     \
                    break;                                                            \
                case NDARRAY_UNOP_LOG10:       p[i] = FN_LOG10(x);                    \
                    break;                                                            \
                case NDARRAY_UNOP_LOGB:        p[i] = FN_LOGB(x);                     \
                    break;                                                            \
                case NDARRAY_UNOP_SINC: {                                             \
                    if (x == (T)0) p[i] = (T)1;                                       \
                    else {                                                            \
                        T px = (T)3.14159265358979323846 * x;                         \
                        p[i] = FN_SIN(px) / px;                                       \
                    }                                                                 \
                    break;                                                            \
                }                                                                     \
                case NDARRAY_UNOP_CLIP:                                               \
                    /* PyTorch clamp(x, lo, hi) = std::min(std::max(x, lo), hi). \
                       cppref:                                                  \
                         std::max(a, b) = (a < b) ? b : a                       \
                         std::min(a, b) = (b < a) ? b : a                       \
                       The branching order matters for NaN: a NaN in `x`        \
                       propagates through the chain (both comparisons are       \
                       false, the NaN-arg is returned), but a NaN in `lo` /     \
                       `hi` is swallowed (the original value survives). Also    \
                       gives the documented PyTorch behaviour when lo > hi:     \
                       result is hi. */                                         \
                    {                                                             \
                        T _y = ((x) < (LO_VAL)) ? (LO_VAL) : (x);                 \
                        p[i] = ((HI_VAL) < _y) ? (HI_VAL) : _y;                   \
                    }                                                             \
                    break;                                                            \
                default: break;                                                       \
            }                                                                         \
        }                                                                             \
    } while (0)

/**
 * @brief CPU floating-point in-place loop for the trig / hyperbolic /
 *        angle-conversion / rounding op family.
 *
 * The 19 ops in this family share a common shape: one libm intrinsic
 * call per element with no extra parameters. The macro is parameterised
 * by the dtype @p T and the libm suffix @p S (`f` for `float`, empty
 * for `double`) so a single body covers both float32 and float64. The
 * `##` token concatenation produces `sin##f → sinf` and `sin## → sin`,
 * matching the dtype-correct libm intrinsic.
 *
 * `degrees` / `radians` are linear; we inline the conversion constant
 * instead of routing through libm. `fix` is numpy's `np.fix` (round
 * toward zero), which equals `trunc` for IEEE 754 floats.
 *
 * @param T       Element type (`float` or `double`).
 * @param S       libm suffix (`f` for float, empty for double).
 * @param OP_TAG  Unary op selector — must be in the trig family.
 */
#define UNARY_TRIG_BODY(T, S, OP_TAG)                                                 \
    do {                                                                              \
        T *p = (T *)data;                                                             \
        const T _deg_factor = (T)(180.0 / 3.14159265358979323846);                    \
        const T _rad_factor = (T)(3.14159265358979323846 / 180.0);                    \
        for (long i = 0; i < n; i++) {                                                \
            T x = p[i];                                                               \
            switch (OP_TAG) {                                                         \
                case NDARRAY_UNOP_SIN:      p[i] = sin##S(x);      break;             \
                case NDARRAY_UNOP_COS:      p[i] = cos##S(x);      break;             \
                case NDARRAY_UNOP_TAN:      p[i] = tan##S(x);      break;             \
                case NDARRAY_UNOP_ARCSIN:   p[i] = asin##S(x);     break;             \
                case NDARRAY_UNOP_ARCCOS:   p[i] = acos##S(x);     break;             \
                case NDARRAY_UNOP_ARCTAN:   p[i] = atan##S(x);     break;             \
                case NDARRAY_UNOP_SINH:     p[i] = sinh##S(x);     break;             \
                case NDARRAY_UNOP_COSH:     p[i] = cosh##S(x);     break;             \
                case NDARRAY_UNOP_TANH:     p[i] = tanh##S(x);     break;             \
                case NDARRAY_UNOP_ARCSINH:  p[i] = asinh##S(x);    break;             \
                case NDARRAY_UNOP_ARCCOSH:  p[i] = acosh##S(x);    break;             \
                case NDARRAY_UNOP_ARCTANH:  p[i] = atanh##S(x);    break;             \
                case NDARRAY_UNOP_DEGREES:  p[i] = x * _deg_factor; break;            \
                case NDARRAY_UNOP_RADIANS:  p[i] = x * _rad_factor; break;            \
                case NDARRAY_UNOP_RINT:     p[i] = rint##S(x);     break;             \
                case NDARRAY_UNOP_FIX:                                                \
                case NDARRAY_UNOP_TRUNC:    p[i] = trunc##S(x);    break;             \
                case NDARRAY_UNOP_FLOOR:    p[i] = floor##S(x);    break;             \
                case NDARRAY_UNOP_CEIL:     p[i] = ceil##S(x);     break;             \
                default: break;                                                       \
            }                                                                         \
        }                                                                             \
    } while (0)

/**
 * @brief Skip leading ASCII whitespace and an optional sign. Returns a
 *        pointer into @p str pointing at the first digit / dot / digit-like
 *        character that the numeric parsers consume.
 */
static const char *unary_skip_sign_ws(const char *str) {
    while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r') str++;
    if (*str == '+' || *str == '-') str++;
    return str;
}

/**
 * @brief Validate that @p str is a syntactically well-formed numeric
 *        literal accepted by `ndarray_set_from_string` for @p dt.
 *
 * Catches the silent acceptance case where `ndarray_set_from_string`
 * would parse "abc" as 0 via `strtod`. The check is strict but cheap:
 * after optional sign / whitespace, require at least one digit and a
 * NUL-terminated suffix (also accepting trailing whitespace).
 *
 * @param[in] str Candidate numeric string (non-NULL).
 * @return 0 on success, -1 if @p str is not a valid numeric literal
 *         (PHP exception in flight).
 */
static int unary_validate_numeric_string(const char *str, const char *which) {
    if (str == NULL || *str == '\0') {
        zend_throw_error(NULL,
            "NDArray clip: '%s' is empty.", which);
        return -1;
    }
    const char *p = unary_skip_sign_ws(str);
    /* Accept inf / infinity / nan tokens (case-insensitive). The token
       must consume the rest of the (trimmed) literal — trailing junk
       such as "infX" / "nanZ" is rejected rather than silently read as
       a valid prefix the way strtod would, mirroring the strict array-
       input inferrer `ndarray_infer_dtype_from_string`. */
    char low3[4] = {0};
    for (int i = 0; i < 3 && p[i]; i++) {
        low3[i] = (char)(p[i] | 0x20);
    }
    if (!strncmp(low3, "inf", 3) || !strncmp(low3, "nan", 3)) {
        const char *t = p + 3;
        if (low3[0] == 'i') {                     /* maybe the "infinity" spelling */
            char low5[6] = {0};
            for (int i = 0; i < 5 && t[i]; i++) low5[i] = (char)(t[i] | 0x20);
            if (!strncmp(low5, "inity", 5)) t += 5;
        }
        while (*t == ' ' || *t == '\t' || *t == '\n' || *t == '\r') t++;
        if (*t != '\0') {
            zend_throw_error(NULL,
                "NDArray clip: '%s' is not a valid number: %s.", which, str);
            return -1;
        }
        return 0;
    }
    int saw_digit = 0;
    while (*p) {
        if (*p >= '0' && *p <= '9') { saw_digit = 1; p++; continue; }
        break;
    }
    /* Optional fractional part. */
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') { saw_digit = 1; p++; }
    }
    /* Optional exponent. */
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        if (!(*p >= '0' && *p <= '9')) {
            zend_throw_error(NULL,
                "NDArray clip: '%s' has malformed exponent: %s.", which, str);
            return -1;
        }
        while (*p >= '0' && *p <= '9') p++;
    }
    /* Trailing whitespace is OK. */
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (!saw_digit || *p != '\0') {
        zend_throw_error(NULL,
            "NDArray clip: '%s' is not a valid number: %s.", which, str);
        return -1;
    }
    return 0;
}

/**
 * @brief Skip leading ASCII whitespace, returning the first non-space char's
 *        pointer. Mirrors `strtoll`'s leading-whitespace handling.
 *
 * @param[in] s NUL-terminated string to scan.
 * @return Pointer into @p s at the first non-whitespace character (the
 *         terminating NUL when @p s is empty or all whitespace).
 */
static inline const char *unary_skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    return s;
}

/** Special-value kind of a validated clip-bound literal. */
typedef enum {
    UNARY_FINITE = 0, UNARY_POS_INF, UNARY_NEG_INF, UNARY_NAN
} unary_special_t;

/**
 * @brief Classify an already-validated clip-bound literal as finite, ±inf,
 *        or nan (case-insensitive, honouring an optional leading sign).
 *
 * @param[in] str NUL-terminated, syntactically validated literal.
 * @return The special-value kind; `UNARY_FINITE` for an ordinary number.
 */
static unary_special_t unary_classify_special(const char *str) {
    const char *p = unary_skip_ws(str);
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
    char low3[4] = {0};
    for (int i = 0; i < 3 && p[i]; i++) low3[i] = (char)(p[i] | 0x20);
    if (!strncmp(low3, "inf", 3)) return neg ? UNARY_NEG_INF : UNARY_POS_INF;
    if (!strncmp(low3, "nan", 3)) return UNARY_NAN;
    return UNARY_FINITE;
}

/**
 * @brief Write the representable extreme of an integer dtype into @p out_buf.
 *
 * Used to give an inf/nan clip bound PyTorch's "no bound" semantics on the
 * 8 integer dtypes (strtoll/strtoull would otherwise read zero digits from
 * the token and yield 0, collapsing the clip range).
 *
 * @param[in]  dt       Canonical dtype string.
 * @param[in]  want_max Non-zero → dtype maximum; zero → dtype minimum
 *                      (0 for the unsigned dtypes).
 * @param[out] out_buf  Buffer of `elsize(dt)` bytes to receive the value.
 * @return 1 when @p dt is one of the 8 integer dtypes (value written);
 *         0 otherwise (a float dtype — caller handles it).
 */
static int unary_write_int_extreme(const char *dt, int want_max, void *out_buf) {
    if (!strcmp(dt, "int8"))   { *(int8_t   *)out_buf = want_max ? INT8_MAX   : INT8_MIN;   return 1; }
    if (!strcmp(dt, "int16"))  { *(int16_t  *)out_buf = want_max ? INT16_MAX  : INT16_MIN;  return 1; }
    if (!strcmp(dt, "int32"))  { *(int32_t  *)out_buf = want_max ? INT32_MAX  : INT32_MIN;  return 1; }
    if (!strcmp(dt, "int64"))  { *(int64_t  *)out_buf = want_max ? INT64_MAX  : INT64_MIN;  return 1; }
    if (!strcmp(dt, "uint8"))  { *(uint8_t  *)out_buf = want_max ? UINT8_MAX  : 0;          return 1; }
    if (!strcmp(dt, "uint16")) { *(uint16_t *)out_buf = want_max ? UINT16_MAX : 0;          return 1; }
    if (!strcmp(dt, "uint32")) { *(uint32_t *)out_buf = want_max ? UINT32_MAX : 0;          return 1; }
    if (!strcmp(dt, "uint64")) { *(uint64_t *)out_buf = want_max ? UINT64_MAX : 0;          return 1; }
    return 0;
}

/**
 * @brief Parse @p str into the typed scalar buffer @p out_buf for @p dt.
 *
 * Validates the string syntactically first so callers get a clean error
 * instead of a silent 0 coerced from a malformed input. For integer
 * dtypes the value is *saturated* to the dtype's representable range
 * (PyTorch `clamp` semantics): a negative bound for an unsigned dtype
 * collapses to 0; a magnitude exceeding the signed dtype's `INT*_MAX`
 * saturates to that max (or `INT*_MIN` if negative); without this
 * saturation, `clip(uint8 tensor, -50, 100)` would silently wrap `-50`
 * via the modulo-2^N cast inside `ndarray_set_from_string`, then see
 * `lo (206) > hi (100)` and collapse every element to `100`. For
 * float dtypes (and `int64`/`uint64`/`float128` where wide-precision
 * strings carry the only loss-free intake), the call falls through to
 * `ndarray_set_from_string` so `strtoll`/`strtoull`/`strtoflt128` keep
 * the full source precision.
 *
 * @param[in]  dt      Canonical dtype string.
 * @param[in]  str     Decimal literal.
 * @param[in]  which   Diagnostic label (`"min"` / `"max"`).
 * @param[out] out_buf Buffer of `elsize(dt)` bytes the value is written into.
 * @return 0 on success, -1 on validation failure (PHP exception in flight).
 */
static int unary_parse_typed_scalar(const char *dt, const char *str,
                                     const char *which, void *out_buf) {
    if (unary_validate_numeric_string(str, which) < 0) return -1;

    /* inf / nan bounds on integer dtypes: strtoll/strtoull read zero
       digits from the token and yield 0, which would collapse the clip
       range. Map the token to the dtype's representable extreme so an
       inf bound acts as PyTorch's "no bound" (−inf → MIN, +inf → MAX),
       and a nan bound becomes the no-op extreme for whichever side it
       sits on (min → MIN, max → MAX), matching how the float path
       silently ignores a nan bound. Float dtypes fall through so strtod
       yields a real ±inf / nan. */
    unary_special_t sp = unary_classify_special(str);
    if (sp != UNARY_FINITE) {
        int want_max = (sp == UNARY_POS_INF) ||
                       (sp == UNARY_NAN && !strcmp(which, "max"));
        if (unary_write_int_extreme(dt, want_max, out_buf)) return 0;
    }

    /* Narrow integer dtypes — saturate the bound to the dtype range so
       out-of-range literals don't wrap via the implicit `(T)strtoll(...)`
       cast inside `ndarray_set_from_string`. int64/uint64 keep the
       wide-precision intake path (their saturating boundary is exactly
       at the strtoll/strtoull edge already). */
    const char *p = unary_skip_ws(str);
    int is_neg = (*p == '-');
    if (!strcmp(dt, "uint8")) {
        if (is_neg) { *(uint8_t *)out_buf = 0; return 0; }
        unsigned long long v = strtoull(p, NULL, 10);
        *(uint8_t *)out_buf = (uint8_t)(v > UINT8_MAX ? UINT8_MAX : v);
        return 0;
    }
    if (!strcmp(dt, "uint16")) {
        if (is_neg) { *(uint16_t *)out_buf = 0; return 0; }
        unsigned long long v = strtoull(p, NULL, 10);
        *(uint16_t *)out_buf = (uint16_t)(v > UINT16_MAX ? UINT16_MAX : v);
        return 0;
    }
    if (!strcmp(dt, "uint32")) {
        if (is_neg) { *(uint32_t *)out_buf = 0; return 0; }
        unsigned long long v = strtoull(p, NULL, 10);
        *(uint32_t *)out_buf = (uint32_t)(v > UINT32_MAX ? UINT32_MAX : v);
        return 0;
    }
    if (!strcmp(dt, "int8")) {
        long long v = strtoll(str, NULL, 10);
        if (v > INT8_MAX)      v = INT8_MAX;
        else if (v < INT8_MIN) v = INT8_MIN;
        *(int8_t *)out_buf = (int8_t)v;
        return 0;
    }
    if (!strcmp(dt, "int16")) {
        long long v = strtoll(str, NULL, 10);
        if (v > INT16_MAX)      v = INT16_MAX;
        else if (v < INT16_MIN) v = INT16_MIN;
        *(int16_t *)out_buf = (int16_t)v;
        return 0;
    }
    if (!strcmp(dt, "int32")) {
        long long v = strtoll(str, NULL, 10);
        if (v > INT32_MAX)      v = INT32_MAX;
        else if (v < INT32_MIN) v = INT32_MIN;
        *(int32_t *)out_buf = (int32_t)v;
        return 0;
    }
    /* uint64: strtoull silently wraps a negative literal modulo 2^64
       (`strtoull("-50")` returns `UINT64_MAX - 49`) — saturate to 0
       explicitly. ERANGE on a positive overflow is what strtoull would
       cap at UINT64_MAX anyway. */
    if (!strcmp(dt, "uint64")) {
        if (is_neg) { *(uint64_t *)out_buf = 0; return 0; }
        /* Fall through to ndarray_set_from_string for the positive path
           so wide-precision literals route through the same parser used
           elsewhere. */
    }
    /* int64 / uint64 (positive) / float* : strtoll / strtoull / strtod /
       strtoflt128 already saturate at the dtype's edge under ERANGE,
       matching the behaviour we want without an explicit upper-bound check. */
    ndarray_set_from_string(dt, (char *)out_buf, 0, str);
    return 0;
}

/**
 * @brief Round every CPU element to @p decimals decimal places using
 *        round-half-to-even, in place.
 *
 * Implements PyTorch's `torch.round(x, decimals=…)` element kernel:
 * `rint(x · 10^decimals) / 10^decimals` for `decimals >= 0`, and the
 * numerically safer `rint(x / 10^|decimals|) · 10^|decimals|` for
 * `decimals < 0` (a positive power of ten is exactly representable for
 * `|decimals| <= 22`, a negative one is not). `rint` honours the default
 * round-to-nearest-even mode, so `round(0.5) == 0` and `round(2.5) == 2`,
 * matching PyTorch and NumPy.
 *
 * Only floating-point compute dtypes reach here — integer inputs are
 * short-circuited to an identity copy by `NDArray_TypedUnaryOp`, and
 * fp4 / fp8 arrive pre-cast to float32.
 *
 * @param[in,out] data     Buffer of @p n elements in dtype @p dt.
 * @param[in]     n        Element count (n > 0).
 * @param[in]     dt       Compute dtype: float16 / float32 / float64 / float128.
 * @param[in]     decimals Decimal places (may be negative).
 * @return 0 on success, -1 if @p dt is not a supported float dtype
 *         (PHP exception in flight).
 */
static int unary_round_cpu_inplace(void *data, long n, const char *dt,
                                   long decimals) {
    int neg = decimals < 0;
    long k = neg ? -decimals : decimals;

    if (!strcmp(dt, NDARRAY_TYPE_FLOAT32)) {
        float *p = (float *)data;
        float pw = powf(10.0f, (float)k);
        for (long i = 0; i < n; i++)
            p[i] = neg ? rintf(p[i] / pw) * pw : rintf(p[i] * pw) / pw;
        return 0;
    }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT64)) {
        double *p = (double *)data;
        double pw = pow(10.0, (double)k);
        for (long i = 0; i < n; i++)
            p[i] = neg ? rint(p[i] / pw) * pw : rint(p[i] * pw) / pw;
        return 0;
    }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT16)) {
        /* Compute through float32 to keep accuracy, mirroring the fp16
           arm of `unary_run_cpu_inplace`. */
        uint16_t *p = (uint16_t *)data;
        float pw = powf(10.0f, (float)k);
        for (long i = 0; i < n; i++) {
            float x = (float)ndarray_fp16_to_double(p[i]);
            float y = neg ? rintf(x / pw) * pw : rintf(x * pw) / pw;
            p[i] = ndarray_double_to_fp16((double)y);
        }
        return 0;
    }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT128)) {
        ndarray_fp128_t *p = (ndarray_fp128_t *)data;
        /* 10^k is exact in fp64 (hence fp128) for k <= 22; for larger k the
           factor is the fp64-rounded power, which matches the GPU dd path so
           CPU and GPU stay in agreement. */
        ndarray_fp128_t pw =
            NDARRAY_FP128_FROM_LD((long double)pow(10.0, (double)k));
        for (long i = 0; i < n; i++) {
            ndarray_fp128_t x = p[i];
            ndarray_fp128_t s = neg ? NDARRAY_FP128_DIV(x, pw)
                                    : NDARRAY_FP128_MUL(x, pw);
            s = NDARRAY_FP128_RINT(s);
            p[i] = neg ? NDARRAY_FP128_MUL(s, pw) : NDARRAY_FP128_DIV(s, pw);
        }
        return 0;
    }
    zend_throw_error(NULL,
        "NDArray_TypedUnaryOp: unsupported round compute dtype \"%s\".", dt);
    return -1;
}

/**
 * @brief Execute the unary op on a CPU buffer in place.
 *
 * @param[in,out] data      Buffer of length @p n × elsize(@p dt) bytes.
 * @param[in]     n         Element count.
 * @param[in]     dt        Computed dtype string (post-promotion).
 * @param[in]     op        Unary op selector.
 * @param[in]     clip_min  Decimal string (clip only); NULL otherwise.
 * @param[in]     clip_max  Decimal string (clip only); NULL otherwise.
 * @param[in]     decimals  Decimal places (round only); ignored otherwise.
 * @return 0 on success, -1 on dispatch error (PHP exception in flight).
 */
static int unary_run_cpu_inplace(void *data, long n, const char *dt,
                                  NDArrayUnaryOp op,
                                  const char *clip_min, const char *clip_max,
                                  long decimals) {
    /* `round` carries a precision argument the generic per-dtype macros
       don't model, so it has its own dtype-correct kernel. */
    if (op == NDARRAY_UNOP_ROUND) {
        return unary_round_cpu_inplace(data, n, dt, decimals);
    }
    /* Pre-parse clip bounds into the right dtype so the inner loop stays
       cheap. The bound buffers are at most 16 bytes (float128). */
    union {
        int8_t   i8;  uint8_t  u8;
        int16_t  i16; uint16_t u16;
        int32_t  i32; uint32_t u32;
        int64_t  i64; uint64_t u64;
        float    f32; double   f64;
        char     fp128[16];
    } lo, hi;
    int is_clip = (op == NDARRAY_UNOP_CLIP);
    if (is_clip) {
        if (unary_parse_typed_scalar(dt, clip_min, "min", &lo) < 0) return -1;
        if (unary_parse_typed_scalar(dt, clip_max, "max", &hi) < 0) return -1;
    } else {
        memset(&lo, 0, sizeof(lo));
        memset(&hi, 0, sizeof(hi));
    }

    if (!strcmp(dt, "int8"))   { UNARY_INT_BODY(int8_t,   uint8_t,  op, lo.i8,  hi.i8);  return 0; }
    if (!strcmp(dt, "uint8"))  { UNARY_INT_BODY(uint8_t,  uint8_t,  op, lo.u8,  hi.u8);  return 0; }
    if (!strcmp(dt, "int16"))  { UNARY_INT_BODY(int16_t,  uint16_t, op, lo.i16, hi.i16); return 0; }
    if (!strcmp(dt, "uint16")) { UNARY_INT_BODY(uint16_t, uint16_t, op, lo.u16, hi.u16); return 0; }
    if (!strcmp(dt, "int32"))  { UNARY_INT_BODY(int32_t,  uint32_t, op, lo.i32, hi.i32); return 0; }
    if (!strcmp(dt, "uint32")) { UNARY_INT_BODY(uint32_t, uint32_t, op, lo.u32, hi.u32); return 0; }
    if (!strcmp(dt, "int64"))  { UNARY_INT_BODY(int64_t,  uint64_t, op, lo.i64, hi.i64); return 0; }
    if (!strcmp(dt, "uint64")) { UNARY_INT_BODY(uint64_t, uint64_t, op, lo.u64, hi.u64); return 0; }

    if (!strcmp(dt, NDARRAY_TYPE_FLOAT32)) {
        if (unary_op_is_trig(op)) {
            UNARY_TRIG_BODY(float, f, op);
        } else {
            UNARY_FLOAT_BODY(float, sqrtf, sinf, fabsf,
                             expf, exp2f, expm1f,
                             logf, log1pf, log2f, log10f, logbf,
                             op, lo.f32, hi.f32);
        }
        return 0;
    }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT64)) {
        if (unary_op_is_trig(op)) {
            UNARY_TRIG_BODY(double, , op);
        } else {
            UNARY_FLOAT_BODY(double, sqrt, sin, fabs,
                             exp, exp2, expm1,
                             log, log1p, log2, log10, logb,
                             op, lo.f64, hi.f64);
        }
        return 0;
    }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT16)) {
        /* Compute through float32 to keep accuracy. */
        uint16_t *p = (uint16_t *)data;
        float lo_f = is_clip ? (float)ndarray_fp16_to_double(lo.u16) : 0.0f;
        float hi_f = is_clip ? (float)ndarray_fp16_to_double(hi.u16) : 0.0f;
        for (long i = 0; i < n; i++) {
            float x = (float)ndarray_fp16_to_double(p[i]);
            float y = x;
            switch (op) {
                case NDARRAY_UNOP_NEGATIVE:    y = -x; break;
                case NDARRAY_UNOP_ABS:         y = fabsf(x); break;
                case NDARRAY_UNOP_POSITIVE:    y = x; break;
                case NDARRAY_UNOP_SIGN:
                    /* NaN-safe sign: propagate NaN per PyTorch. */
                    y = (x != x) ? x : (float)((x > 0.0f) - (x < 0.0f));
                    break;
                case NDARRAY_UNOP_SQUARE:      y = x * x; break;
                case NDARRAY_UNOP_RECIPROCAL:  y = 1.0f / x; break;
                case NDARRAY_UNOP_SQRT:        y = sqrtf(x); break;
                case NDARRAY_UNOP_RSQRT:       y = 1.0f / sqrtf(x); break;
                case NDARRAY_UNOP_EXP:         y = expf(x); break;
                case NDARRAY_UNOP_EXP2:        y = exp2f(x); break;
                case NDARRAY_UNOP_EXPM1:       y = expm1f(x); break;
                case NDARRAY_UNOP_LOG:         y = logf(x); break;
                case NDARRAY_UNOP_LOG1P:       y = log1pf(x); break;
                case NDARRAY_UNOP_LOG2:        y = log2f(x); break;
                case NDARRAY_UNOP_LOG10:       y = log10f(x); break;
                case NDARRAY_UNOP_LOGB:        y = logbf(x); break;
                case NDARRAY_UNOP_SINC:
                    if (x == 0.0f) y = 1.0f;
                    else { float px = 3.14159265358979323846f * x; y = sinf(px) / px; }
                    break;
                /* Trigonometric / hyperbolic / angle / rounding via float32. */
                case NDARRAY_UNOP_SIN:      y = sinf(x);    break;
                case NDARRAY_UNOP_COS:      y = cosf(x);    break;
                case NDARRAY_UNOP_TAN:      y = tanf(x);    break;
                case NDARRAY_UNOP_ARCSIN:   y = asinf(x);   break;
                case NDARRAY_UNOP_ARCCOS:   y = acosf(x);   break;
                case NDARRAY_UNOP_ARCTAN:   y = atanf(x);   break;
                case NDARRAY_UNOP_SINH:     y = sinhf(x);   break;
                case NDARRAY_UNOP_COSH:     y = coshf(x);   break;
                case NDARRAY_UNOP_TANH:     y = tanhf(x);   break;
                case NDARRAY_UNOP_ARCSINH:  y = asinhf(x);  break;
                case NDARRAY_UNOP_ARCCOSH:  y = acoshf(x);  break;
                case NDARRAY_UNOP_ARCTANH:  y = atanhf(x);  break;
                case NDARRAY_UNOP_DEGREES:  y = x * (float)(180.0 / 3.14159265358979323846); break;
                case NDARRAY_UNOP_RADIANS:  y = x * (float)(3.14159265358979323846 / 180.0); break;
                case NDARRAY_UNOP_RINT:     y = rintf(x);   break;
                case NDARRAY_UNOP_FIX:
                case NDARRAY_UNOP_TRUNC:    y = truncf(x);  break;
                case NDARRAY_UNOP_FLOOR:    y = floorf(x);  break;
                case NDARRAY_UNOP_CEIL:     y = ceilf(x);   break;
                case NDARRAY_UNOP_CLIP: {
                    /* min(max(x, lo), hi) per PyTorch clamp; see UNARY_FLOAT_BODY. */
                    float _yc = (x < lo_f) ? lo_f : x;
                    y = (hi_f < _yc) ? hi_f : _yc;
                    break;
                }
                default: break;
            }
            p[i] = ndarray_double_to_fp16((double)y);
        }
        return 0;
    }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT128)) {
        ndarray_fp128_t *p = (ndarray_fp128_t *)data;
        ndarray_fp128_t lo_v = NDARRAY_FP128_ZERO(), hi_v = NDARRAY_FP128_ZERO();
        if (is_clip) {
            memcpy(&lo_v, lo.fp128, sizeof(lo_v));
            memcpy(&hi_v, hi.fp128, sizeof(hi_v));
        }
        for (long i = 0; i < n; i++) {
            ndarray_fp128_t x = p[i];
            ndarray_fp128_t y = x;
            switch (op) {
                case NDARRAY_UNOP_NEGATIVE:    y = NDARRAY_FP128_NEG(x); break;
                case NDARRAY_UNOP_ABS:         y = NDARRAY_FP128_ABS(x); break;
                case NDARRAY_UNOP_POSITIVE:    y = x; break;
                case NDARRAY_UNOP_SIGN: {
                    /* PyTorch sign(NaN) = NaN. Without the explicit NaN
                       guard the else branch below would silently return
                       +1 (LT(NaN, 0) is false → falls through). */
                    if (NDARRAY_FP128_ISNAN(x))       y = x;
                    else if (NDARRAY_FP128_ISZERO(x)) y = NDARRAY_FP128_ZERO();
                    else if (NDARRAY_FP128_LT(x, NDARRAY_FP128_ZERO()))
                        y = NDARRAY_FP128_NEG(NDARRAY_FP128_ONE());
                    else y = NDARRAY_FP128_ONE();
                    break;
                }
                case NDARRAY_UNOP_SQUARE:      y = NDARRAY_FP128_MUL(x, x); break;
                case NDARRAY_UNOP_RECIPROCAL:  y = NDARRAY_FP128_DIV(NDARRAY_FP128_ONE(), x); break;
                case NDARRAY_UNOP_SQRT:        y = NDARRAY_FP128_SQRT(x); break;
                case NDARRAY_UNOP_RSQRT:
                    y = NDARRAY_FP128_DIV(NDARRAY_FP128_ONE(), NDARRAY_FP128_SQRT(x));
                    break;
                case NDARRAY_UNOP_EXP:    y = NDARRAY_FP128_EXP(x);    break;
                case NDARRAY_UNOP_EXP2:   y = NDARRAY_FP128_EXP2(x);   break;
                case NDARRAY_UNOP_EXPM1:  y = NDARRAY_FP128_EXPM1(x);  break;
                case NDARRAY_UNOP_LOG:    y = NDARRAY_FP128_LOG(x);    break;
                case NDARRAY_UNOP_LOG1P:  y = NDARRAY_FP128_LOG1P(x);  break;
                case NDARRAY_UNOP_LOG2:   y = NDARRAY_FP128_LOG2(x);   break;
                case NDARRAY_UNOP_LOG10:  y = NDARRAY_FP128_LOG10(x);  break;
                case NDARRAY_UNOP_LOGB:   y = NDARRAY_FP128_LOGB(x);   break;
                case NDARRAY_UNOP_SIN:     y = NDARRAY_FP128_SIN(x);     break;
                case NDARRAY_UNOP_COS:     y = NDARRAY_FP128_COS(x);     break;
                case NDARRAY_UNOP_TAN:     y = NDARRAY_FP128_TAN(x);     break;
                case NDARRAY_UNOP_ARCSIN:  y = NDARRAY_FP128_ARCSIN(x);  break;
                case NDARRAY_UNOP_ARCCOS:  y = NDARRAY_FP128_ARCCOS(x);  break;
                case NDARRAY_UNOP_ARCTAN:  y = NDARRAY_FP128_ARCTAN(x);  break;
                case NDARRAY_UNOP_SINH:    y = NDARRAY_FP128_SINH(x);    break;
                case NDARRAY_UNOP_COSH:    y = NDARRAY_FP128_COSH(x);    break;
                case NDARRAY_UNOP_TANH:    y = NDARRAY_FP128_TANH(x);    break;
                case NDARRAY_UNOP_ARCSINH: y = NDARRAY_FP128_ARCSINH(x); break;
                case NDARRAY_UNOP_ARCCOSH: y = NDARRAY_FP128_ARCCOSH(x); break;
                case NDARRAY_UNOP_ARCTANH: y = NDARRAY_FP128_ARCTANH(x); break;
                case NDARRAY_UNOP_DEGREES:
                    /* libquadmath path: M_PIq has full 113-bit pi; DD path
                       gets ~64 bits via long-double — both adequate for the
                       linear conversion x · 180/π. */
                    y = NDARRAY_FP128_MUL(x, NDARRAY_FP128_FROM_LD(
                            (long double)(180.0L / 3.14159265358979323846L)));
                    break;
                case NDARRAY_UNOP_RADIANS:
                    y = NDARRAY_FP128_MUL(x, NDARRAY_FP128_FROM_LD(
                            (long double)(3.14159265358979323846L / 180.0L)));
                    break;
                case NDARRAY_UNOP_RINT:    y = NDARRAY_FP128_RINT(x);    break;
                case NDARRAY_UNOP_FIX:
                case NDARRAY_UNOP_TRUNC:   y = NDARRAY_FP128_TRUNC(x);   break;
                case NDARRAY_UNOP_FLOOR:   y = NDARRAY_FP128_FLOOR(x);   break;
                case NDARRAY_UNOP_CEIL:    y = NDARRAY_FP128_CEIL(x);    break;
                case NDARRAY_UNOP_SINC: {
                    if (NDARRAY_FP128_ISZERO(x)) y = NDARRAY_FP128_ONE();
                    else {
                        /* sin works at full fp128 precision on libquadmath
                           (`sinq`); on the DD fallback it routes through
                           double - same accuracy bound as the GPU dd path.
                           `M_PIq` from quadmath gives the full 34-digit pi
                           when available, otherwise we promote a long-double
                           literal (~19 digits on x86_64). */
#if NDARRAY_HAVE_FLOAT128 && HAVE_QUADMATH
                        ndarray_fp128_t pi = M_PIq;
#else
                        ndarray_fp128_t pi = NDARRAY_FP128_FROM_LD(
                            3.141592653589793238462643383279502884L);
#endif
                        ndarray_fp128_t px = NDARRAY_FP128_MUL(pi, x);
                        ndarray_fp128_t s  = NDARRAY_FP128_SIN(px);
                        y = NDARRAY_FP128_DIV(s, px);
                    }
                    break;
                }
                case NDARRAY_UNOP_CLIP: {
                    /* min(max(x, lo), hi) per PyTorch clamp; see UNARY_FLOAT_BODY. */
                    ndarray_fp128_t _yc = NDARRAY_FP128_LT(x, lo_v) ? lo_v : x;
                    y = NDARRAY_FP128_LT(hi_v, _yc) ? hi_v : _yc;
                    break;
                }
                default: break;
            }
            /* NaN-sign canonicalization happens at stringification time
               (`ndarray_fp128_to_string`) rather than here. This keeps
               the in-memory bit pattern mathematically faithful:
               `NumPower::negative(NaN)` flips the sign bit (matches
               NumPy / PyTorch `neg` on NaN), `NumPower::positive(NaN)`
               preserves the input, while `__toString` / `toArray`
               render every NaN as the unsigned `"nan"` literal across
               every fp dtype. */
            p[i] = y;
        }
        return 0;
    }
    zend_throw_error(NULL,
        "NDArray_TypedUnaryOp: unsupported compute dtype \"%s\".", dt);
    return -1;
}

#ifdef HAVE_CUBLAS
/**
 * @brief Round a typed GPU buffer to @p decimals places in place,
 *        round-half-to-even, matching `unary_round_cpu_inplace`.
 *
 * Routes to the per-dtype `cuda_round_*` wrapper, which computes the
 * `10^|decimals|` factor on the host and applies `rint(x · f) / f`
 * (decimals ≥ 0) or `rint(x / f) · f` (decimals < 0) on device. Only
 * floating-point compute dtypes reach here (integers are short-circuited
 * to a copy upstream; fp4 / fp8 arrive pre-cast to float32).
 *
 * @param[in,out] data     Device buffer of @p n elements in dtype @p dt.
 * @param[in]     n        Element count (n > 0).
 * @param[in]     dt       Compute dtype: float16 / float32 / float64 / float128.
 * @param[in]     decimals Decimal places (may be negative).
 * @return 0 on success, -1 if @p dt is not a supported float dtype
 *         (PHP exception in flight).
 */
static int unary_round_gpu_inplace(void *data, long n, const char *dt,
                                   long decimals) {
    int ni = (int)n;
    int dec = (int)decimals;
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT32)) { cuda_round_f32((float    *)data, dec, ni); return 0; }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT64)) { cuda_round_f64((double   *)data, dec, ni); return 0; }
    if (!strcmp(dt, NDARRAY_TYPE_FLOAT16)) { cuda_round_f16((uint16_t *)data, dec, ni); return 0; }
    if (!strcmp(dt, "float128"))           { cuda_round_dd ((double   *)data, dec, ni); return 0; }
    zend_throw_error(NULL,
        "NDArray_TypedUnaryOp GPU: unsupported round compute dtype \"%s\".", dt);
    return -1;
}

/**
 * @brief Dispatch the unary op against a typed GPU buffer in place.
 *
 * Routes to the right `cuda_<op>_<tag>` wrapper for every supported
 * compute dtype. Caller has already cast the input to the compute
 * dtype on GPU via `NDArray_AsType` / `NDArray_Copy`.
 *
 * `clip_min` / `clip_max` are decimal strings parsed losslessly into
 * the typed bound for the clip op; `decimals` is the precision for the
 * round op (ignored by every other op).
 */
static int unary_run_gpu_inplace(void *data, long n, const char *dt,
                                  NDArrayUnaryOp op,
                                  const char *clip_min, const char *clip_max,
                                  long decimals) {
    /* `round` carries a precision argument; dispatch to its dedicated
       per-dtype kernels rather than the generic `UNOP_GPU_*` blocks. */
    if (op == NDARRAY_UNOP_ROUND) {
        return unary_round_gpu_inplace(data, n, dt, decimals);
    }
    union {
        int8_t   i8;  uint8_t  u8;
        int16_t  i16; uint16_t u16;
        int32_t  i32; uint32_t u32;
        int64_t  i64; uint64_t u64;
        float    f32; double   f64;
        char     fp128[16];
    } lo, hi;
    int is_clip = (op == NDARRAY_UNOP_CLIP);
    if (is_clip) {
        if (unary_parse_typed_scalar(dt, clip_min, "min", &lo) < 0) return -1;
        if (unary_parse_typed_scalar(dt, clip_max, "max", &hi) < 0) return -1;
    } else {
        memset(&lo, 0, sizeof(lo));
        memset(&hi, 0, sizeof(hi));
    }
    int ni = (int)n;

    /* Transcendental dispatch (exp / exp2 / expm1 / log / log1p / log2 /
       log10 / logb) — runs first because the existing `UNOP_GPU_DT`
       macro's `default:` arm throws on unknown ops, which would block
       us from extending it without re-typing every dtype line. Integer
       inputs are promoted to fp32 / fp64 upstream by
       `unary_result_dtype`, so only floating-point compute dtypes
       reach this block. fp128 is handled in its own block below. */
#define UNOP_GPU_TRANSC_DT(DTSTR, T, EXP, EXP2, EXPM1, LOG, LOG1P, LOG2, LOG10, LOGB) \
    if (!strcmp(dt, DTSTR)) {                                                          \
        T *p = (T *)data;                                                              \
        switch (op) {                                                                  \
            case NDARRAY_UNOP_EXP:   EXP  (p, ni); return 0;                           \
            case NDARRAY_UNOP_EXP2:  EXP2 (p, ni); return 0;                           \
            case NDARRAY_UNOP_EXPM1: EXPM1(p, ni); return 0;                           \
            case NDARRAY_UNOP_LOG:   LOG  (p, ni); return 0;                           \
            case NDARRAY_UNOP_LOG1P: LOG1P(p, ni); return 0;                           \
            case NDARRAY_UNOP_LOG2:  LOG2 (p, ni); return 0;                           \
            case NDARRAY_UNOP_LOG10: LOG10(p, ni); return 0;                           \
            case NDARRAY_UNOP_LOGB:  LOGB (p, ni); return 0;                           \
            default: break; /* fall through to UNOP_GPU_DT */                          \
        }                                                                              \
    }
    UNOP_GPU_TRANSC_DT("float32", float,
        cuda_exp_f32, cuda_exp2_f32, cuda_expm1_f32,
        cuda_log_f32, cuda_log1p_f32, cuda_log2_f32, cuda_log10_f32, cuda_logb_f32)
    UNOP_GPU_TRANSC_DT("float64", double,
        cuda_exp_f64, cuda_exp2_f64, cuda_expm1_f64,
        cuda_log_f64, cuda_log1p_f64, cuda_log2_f64, cuda_log10_f64, cuda_logb_f64)
    UNOP_GPU_TRANSC_DT("float16", uint16_t,
        cuda_exp_f16, cuda_exp2_f16, cuda_expm1_f16,
        cuda_log_f16, cuda_log1p_f16, cuda_log2_f16, cuda_log10_f16, cuda_logb_f16)
#undef UNOP_GPU_TRANSC_DT

    /* Trig / hyperbolic / angle / rounding dispatch — same fall-through
       contract as the transcendental block: switch on op, fall through
       to `UNOP_GPU_DT` only for ops outside this family. Token-pasting
       the dtype suffix (`f32` / `f64` / `f16`) into the wrapper name
       keeps the 19-op block legible without a 20-arg macro. */
#define UNOP_GPU_TRIG_DT(DTSTR, T, S)                                                  \
    if (!strcmp(dt, DTSTR)) {                                                          \
        T *p = (T *)data;                                                              \
        switch (op) {                                                                  \
            case NDARRAY_UNOP_SIN:      cuda_sin_##S      (p, ni); return 0;           \
            case NDARRAY_UNOP_COS:      cuda_cos_##S      (p, ni); return 0;           \
            case NDARRAY_UNOP_TAN:      cuda_tan_##S      (p, ni); return 0;           \
            case NDARRAY_UNOP_ARCSIN:   cuda_arcsin_##S   (p, ni); return 0;           \
            case NDARRAY_UNOP_ARCCOS:   cuda_arccos_##S   (p, ni); return 0;           \
            case NDARRAY_UNOP_ARCTAN:   cuda_arctan_##S   (p, ni); return 0;           \
            case NDARRAY_UNOP_SINH:     cuda_sinh_##S     (p, ni); return 0;           \
            case NDARRAY_UNOP_COSH:     cuda_cosh_##S     (p, ni); return 0;           \
            case NDARRAY_UNOP_TANH:     cuda_tanh_##S     (p, ni); return 0;           \
            case NDARRAY_UNOP_ARCSINH:  cuda_arcsinh_##S  (p, ni); return 0;           \
            case NDARRAY_UNOP_ARCCOSH:  cuda_arccosh_##S  (p, ni); return 0;           \
            case NDARRAY_UNOP_ARCTANH:  cuda_arctanh_##S  (p, ni); return 0;           \
            case NDARRAY_UNOP_DEGREES:  cuda_degrees_##S  (p, ni); return 0;           \
            case NDARRAY_UNOP_RADIANS:  cuda_radians_##S  (p, ni); return 0;           \
            case NDARRAY_UNOP_RINT:     cuda_rint_##S     (p, ni); return 0;           \
            case NDARRAY_UNOP_FIX:                                                     \
            case NDARRAY_UNOP_TRUNC:    cuda_trunc_##S    (p, ni); return 0;           \
            case NDARRAY_UNOP_FLOOR:    cuda_floor_##S    (p, ni); return 0;           \
            case NDARRAY_UNOP_CEIL:     cuda_ceil_##S     (p, ni); return 0;           \
            default: break;                                                            \
        }                                                                              \
    }
    UNOP_GPU_TRIG_DT("float32", float,    f32)
    UNOP_GPU_TRIG_DT("float64", double,   f64)
    UNOP_GPU_TRIG_DT("float16", uint16_t, f16)
#undef UNOP_GPU_TRIG_DT

#define UNOP_GPU_DT(DTSTR, T, NEG, ABS, SIGN, RECIP, SQRT, RSQRT, SQUARE, SINC, CLIP) \
    if (!strcmp(dt, DTSTR)) {                                                          \
        T *p = (T *)data;                                                              \
        switch (op) {                                                                  \
            case NDARRAY_UNOP_NEGATIVE:                NEG (p, ni); break;             \
            case NDARRAY_UNOP_ABS:                    {ABS;}                  break;   \
            case NDARRAY_UNOP_POSITIVE:                /* in-place no-op */   break;   \
            case NDARRAY_UNOP_SIGN:                    SIGN(p, ni); break;             \
            case NDARRAY_UNOP_RECIPROCAL:             {RECIP;}                break;   \
            case NDARRAY_UNOP_SQRT:                   {SQRT;}                 break;   \
            case NDARRAY_UNOP_RSQRT:                  {RSQRT;}                break;   \
            case NDARRAY_UNOP_SQUARE:                  SQUARE(p, ni); break;           \
            case NDARRAY_UNOP_SINC:                   {SINC;}                 break;   \
            case NDARRAY_UNOP_CLIP:                    CLIP; break;                    \
            default:                                                                   \
                zend_throw_error(NULL,                                                 \
                    "NDArray_TypedUnaryOp GPU: unsupported op for dtype %s.", DTSTR);  \
                return -1;                                                             \
        }                                                                              \
        return 0;                                                                      \
    }

    /* Helpers for "not supported on integer dtype" — the dispatcher
       promotes ints to float before launch for these ops, so reaching
       these branches is a programmer error. */
#define UNOP_INT_NOT_FLOAT { zend_throw_error(NULL,                                    \
        "NDArray_TypedUnaryOp GPU: integer dispatched to float-only op"); return -1; }

    UNOP_GPU_DT("int8", int8_t,
        cuda_negate_i8, cuda_abs_i8(p, ni), cuda_sign_i8,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_i8, UNOP_INT_NOT_FLOAT,
        cuda_clip_i8(p, lo.i8, hi.i8, ni))
    UNOP_GPU_DT("uint8", uint8_t,
        cuda_negate_u8, /* uint abs is a no-op */ (void)0, cuda_sign_u8,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_u8, UNOP_INT_NOT_FLOAT,
        cuda_clip_u8(p, lo.u8, hi.u8, ni))
    UNOP_GPU_DT("int16", int16_t,
        cuda_negate_i16, cuda_abs_i16(p, ni), cuda_sign_i16,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_i16, UNOP_INT_NOT_FLOAT,
        cuda_clip_i16(p, lo.i16, hi.i16, ni))
    UNOP_GPU_DT("uint16", uint16_t,
        cuda_negate_u16, (void)0, cuda_sign_u16,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_u16, UNOP_INT_NOT_FLOAT,
        cuda_clip_u16(p, lo.u16, hi.u16, ni))
    UNOP_GPU_DT("int32", int32_t,
        cuda_negate_i32, cuda_abs_i32(p, ni), cuda_sign_i32,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_i32, UNOP_INT_NOT_FLOAT,
        cuda_clip_i32(p, lo.i32, hi.i32, ni))
    UNOP_GPU_DT("uint32", uint32_t,
        cuda_negate_u32, (void)0, cuda_sign_u32,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_u32, UNOP_INT_NOT_FLOAT,
        cuda_clip_u32(p, lo.u32, hi.u32, ni))
    UNOP_GPU_DT("int64", int64_t,
        cuda_negate_i64, cuda_abs_i64(p, ni), cuda_sign_i64,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_i64, UNOP_INT_NOT_FLOAT,
        cuda_clip_i64(p, lo.i64, hi.i64, ni))
    UNOP_GPU_DT("uint64", uint64_t,
        cuda_negate_u64, (void)0, cuda_sign_u64,
        UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT, UNOP_INT_NOT_FLOAT,
        cuda_square_u64, UNOP_INT_NOT_FLOAT,
        cuda_clip_u64(p, lo.u64, hi.u64, ni))
    UNOP_GPU_DT("float32", float,
        cuda_negate_f32, cuda_abs_f32(p, ni), cuda_sign_f32,
        cuda_recip_f32(p, ni), cuda_sqrt_f32(p, ni), cuda_rsqrt_f32(p, ni),
        cuda_square_f32, cuda_sinc_f32(p, ni),
        cuda_clip_f32(p, lo.f32, hi.f32, ni))
    UNOP_GPU_DT("float64", double,
        cuda_negate_f64, cuda_abs_f64(p, ni), cuda_sign_f64,
        cuda_recip_f64(p, ni), cuda_sqrt_f64(p, ni), cuda_rsqrt_f64(p, ni),
        cuda_square_f64, cuda_sinc_f64(p, ni),
        cuda_clip_f64(p, lo.f64, hi.f64, ni))
    UNOP_GPU_DT("float16", uint16_t,
        cuda_negate_f16, cuda_abs_f16(p, ni), cuda_sign_f16,
        cuda_recip_f16(p, ni), cuda_sqrt_f16(p, ni), cuda_rsqrt_f16(p, ni),
        cuda_square_f16,  cuda_sinc_f16(p, ni),
        cuda_clip_f16(p, (float)ndarray_fp16_to_double(lo.u16),
                          (float)ndarray_fp16_to_double(hi.u16), ni))

#undef UNOP_GPU_DT
#undef UNOP_INT_NOT_FLOAT

    if (!strcmp(dt, "float128")) {
        double *p = (double *)data;
        double lo_hi = 0.0, lo_lo = 0.0, hi_hi = 0.0, hi_lo = 0.0;
        if (is_clip) {
            ndarray_fp128_t v;
            memcpy(&v, lo.fp128, sizeof(v));
#if NDARRAY_HAVE_FLOAT128
            lo_hi = (double)v;
            lo_lo = (double)(v - (ndarray_fp128_t)lo_hi);
#else
            lo_hi = v.hi; lo_lo = v.lo;
#endif
            memcpy(&v, hi.fp128, sizeof(v));
#if NDARRAY_HAVE_FLOAT128
            hi_hi = (double)v;
            hi_lo = (double)(v - (ndarray_fp128_t)hi_hi);
#else
            hi_hi = v.hi; hi_lo = v.lo;
#endif
        }
        switch (op) {
            case NDARRAY_UNOP_NEGATIVE:   cuda_negate_dd(p, ni); break;
            case NDARRAY_UNOP_ABS:        cuda_abs_dd  (p, ni); break;
            case NDARRAY_UNOP_POSITIVE:                          break;
            case NDARRAY_UNOP_SIGN:       cuda_sign_dd (p, ni); break;
            case NDARRAY_UNOP_RECIPROCAL: cuda_recip_dd(p, ni); break;
            case NDARRAY_UNOP_SQRT:       cuda_sqrt_dd (p, ni); break;
            case NDARRAY_UNOP_RSQRT:      cuda_rsqrt_dd(p, ni); break;
            case NDARRAY_UNOP_SQUARE:     cuda_square_dd(p, ni); break;
            case NDARRAY_UNOP_SINC:       cuda_sinc_dd (p, ni); break;
            case NDARRAY_UNOP_EXP:        cuda_exp_dd   (p, ni); break;
            case NDARRAY_UNOP_EXP2:       cuda_exp2_dd  (p, ni); break;
            case NDARRAY_UNOP_EXPM1:      cuda_expm1_dd (p, ni); break;
            case NDARRAY_UNOP_LOG:        cuda_log_dd   (p, ni); break;
            case NDARRAY_UNOP_LOG1P:      cuda_log1p_dd (p, ni); break;
            case NDARRAY_UNOP_LOG2:       cuda_log2_dd  (p, ni); break;
            case NDARRAY_UNOP_LOG10:      cuda_log10_dd (p, ni); break;
            case NDARRAY_UNOP_LOGB:       cuda_logb_dd  (p, ni); break;
            case NDARRAY_UNOP_SIN:       cuda_sin_dd     (p, ni); break;
            case NDARRAY_UNOP_COS:       cuda_cos_dd     (p, ni); break;
            case NDARRAY_UNOP_TAN:       cuda_tan_dd     (p, ni); break;
            case NDARRAY_UNOP_ARCSIN:    cuda_arcsin_dd  (p, ni); break;
            case NDARRAY_UNOP_ARCCOS:    cuda_arccos_dd  (p, ni); break;
            case NDARRAY_UNOP_ARCTAN:    cuda_arctan_dd  (p, ni); break;
            case NDARRAY_UNOP_SINH:      cuda_sinh_dd    (p, ni); break;
            case NDARRAY_UNOP_COSH:      cuda_cosh_dd    (p, ni); break;
            case NDARRAY_UNOP_TANH:      cuda_tanh_dd    (p, ni); break;
            case NDARRAY_UNOP_ARCSINH:   cuda_arcsinh_dd (p, ni); break;
            case NDARRAY_UNOP_ARCCOSH:   cuda_arccosh_dd (p, ni); break;
            case NDARRAY_UNOP_ARCTANH:   cuda_arctanh_dd (p, ni); break;
            case NDARRAY_UNOP_DEGREES:   cuda_degrees_dd (p, ni); break;
            case NDARRAY_UNOP_RADIANS:   cuda_radians_dd (p, ni); break;
            case NDARRAY_UNOP_RINT:      cuda_rint_dd    (p, ni); break;
            case NDARRAY_UNOP_FIX:
            case NDARRAY_UNOP_TRUNC:     cuda_trunc_dd   (p, ni); break;
            case NDARRAY_UNOP_FLOOR:     cuda_floor_dd   (p, ni); break;
            case NDARRAY_UNOP_CEIL:      cuda_ceil_dd    (p, ni); break;
            case NDARRAY_UNOP_CLIP:
                cuda_clip_dd(p, lo_hi, lo_lo, hi_hi, hi_lo, ni); break;
            default:
                zend_throw_error(NULL,
                    "NDArray_TypedUnaryOp GPU: unsupported op for fp128.");
                return -1;
        }
        return 0;
    }
    zend_throw_error(NULL,
        "NDArray_TypedUnaryOp GPU: unsupported compute dtype \"%s\".", dt);
    return -1;
}
#endif /* HAVE_CUBLAS */

NDArray *
NDArray_TypedUnaryOp(NDArrayUnaryOp op, NDArray *nda,
                     const char *clip_min, const char *clip_max,
                     long round_decimals) {
    if (nda == NULL) return NULL;

    /* round(x, 0) is bit-identical to rint(x) on every dtype and device,
       so reuse the existing rint path (the integer short-circuit and the
       dedicated rint kernels). With a non-zero precision the op stays
       ROUND and flows through `unary_round_*_inplace`. */
    if (op == NDARRAY_UNOP_ROUND && round_decimals == 0) {
        op = NDARRAY_UNOP_RINT;
    }

    const char *input_dt   = NDArray_TYPE(nda);
    const char *result_dt  = unary_result_dtype(op, input_dt);
    const char *compute_dt = unary_compute_dtype(result_dt);
    int device             = NDArray_DEVICE(nda);

    /* Empty input → empty output of the right dtype. The kernel doesn't
       handle n == 0 differently, but skipping the launch keeps the GPU
       path leak-free and matches NumPy. */
    if (NDArray_NUMELEMENTS(nda) == 0) {
        int ndim = NDArray_NDIM(nda);
        int *shape = emalloc(sizeof(int) * (ndim > 0 ? ndim : 1));
        if (ndim > 0) memcpy(shape, NDArray_SHAPE(nda), sizeof(int) * ndim);
        else          shape[0] = 1;
        return NDArray_Empty(shape, ndim, result_dt, device);
    }

    /* Rounding ops on integer dtypes are identity: an integer is already
       its own rounded value (positive/negative). Short-circuit to a
       device-preserving NDArray_Copy so we never launch a no-op kernel
       — `unary_run_gpu_inplace` doesn't dispatch integer cases for the
       rounding family (they have no work to do on the GPU side either).
       This applies even on CPU: skipping the loop saves a pass and
       keeps the contract uniform. `round` with a non-zero precision is
       included here too: NumPower's rounding family preserves integer
       dtypes (an integer is already integer-valued), so it returns the
       input unchanged for any precision — cast to a float dtype to round
       integers to negative decimal places. (round with precision 0 was
       already rewritten to RINT above.) */
    if (unary_is_int_dtype(input_dt) &&
        (op == NDARRAY_UNOP_RINT  || op == NDARRAY_UNOP_FIX   ||
         op == NDARRAY_UNOP_TRUNC || op == NDARRAY_UNOP_FLOOR ||
         op == NDARRAY_UNOP_CEIL  || op == NDARRAY_UNOP_ROUND)) {
        return NDArray_Copy(nda, device);
    }

    /* Stage the result on the same device as the input. Two cases:
       (a) compute_dt == input_dt (every dtype except fp4 / fp8): clone
           the input buffer via NDArray_Copy, then run the op in place
           on the copy. NDArray_Copy stays on-device on every dtype.
       (b) compute_dt != input_dt (fp4 / fp8 inputs only): NDArray_AsType
           casts to float32 on the input's device. GPU AsType keeps the
           fp4 / fp8 → fp32 cast on GPU (`cuda_cast_fp4_to_f32` /
           `cuda_cast_fp8_to_f32`); CPU AsType uses the generic
           element-wise loop. fp128 is never the source of a promotion
           because (a) covers it. */
    NDArray *work = is_type(input_dt, compute_dt)
        ? NDArray_Copy  (nda, device)
        : NDArray_AsType(nda, compute_dt);
    if (work == NULL) return NULL;

    long n = NDArray_NUMELEMENTS(work);
    int rc;
    if (device == NDARRAY_DEVICE_CPU) {
        rc = unary_run_cpu_inplace(NDArray_DATA(work), n, compute_dt,
                                   op, clip_min, clip_max, round_decimals);
    } else {
#ifdef HAVE_CUBLAS
        rc = unary_run_gpu_inplace(NDArray_DATA(work), n, compute_dt,
                                   op, clip_min, clip_max, round_decimals);
#else
        zend_throw_error(NULL,
            "NDArray_TypedUnaryOp: GPU input but extension built without CUDA.");
        rc = -1;
#endif
    }
    if (rc != 0) {
        NDArray_FREE(work);
        return NULL;
    }

    /* If we computed in a wider dtype (fp4 / fp8 → float32), cast back
       to the declared result dtype now. NDArray_AsType stays on-device
       for those narrow-float pairs. */
    if (!is_type(compute_dt, result_dt)) {
        NDArray *narrow = NDArray_AsType(work, result_dt);
        NDArray_FREE(work);
        return narrow;
    }
    return work;
}
