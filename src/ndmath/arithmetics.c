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

    /* fp4 / fp8 widened dtype is still themselves (float family kept as-is);
       no native binop, so stage through float16. */
    if (!strcmp(out_dt, "float4") || !strcmp(out_dt, "float8")) {
        int *cast_shape = emalloc(sizeof(int) * (out_ndim > 0 ? out_ndim : 1));
        if (out_ndim > 0) {
            memcpy(cast_shape, out_shape, sizeof(int) * out_ndim);
        } else {
            cast_shape[0] = 1;
        }
        NDArray *out_f16 = NDArray_Empty(cast_shape, out_ndim, "float16",
                                          NDARRAY_DEVICE_GPU);
        if (out_f16 == NULL) {
            if (a_rolled != a_widened) NDArray_FREE(a_rolled);
            if (a_widened != a) NDArray_FREE(a_widened);
            NDArray_FREE(out);
            return NULL;
        }
        int *src_shape = emalloc(sizeof(int));
        src_shape[0] = (int)n_per_slice;
        NDArray *slice_f16 = NDArray_Empty(src_shape, 1, "float16",
                                            NDARRAY_DEVICE_GPU);
        if (slice_f16 == NULL) {
            if (a_rolled != a_widened) NDArray_FREE(a_rolled);
            if (a_widened != a) NDArray_FREE(a_widened);
            NDArray_FREE(out_f16);
            NDArray_FREE(out);
            return NULL;
        }
        for (long k = 0; k < s_axis; k++) {
            const uint8_t *src_k = (const uint8_t *)(src_data + (size_t)k * slice_bytes);
            if (!strcmp(out_dt, "float4")) {
                cuda_cast_fp4_to_f16((uint8_t *)src_k,
                                     (uint16_t *)NDArray_DATA(slice_f16), n_slice_i);
            } else {
                cuda_cast_fp8_to_f16((uint8_t *)src_k,
                                     (uint16_t *)NDArray_DATA(slice_f16), n_slice_i);
            }
            if (k == 0) {
                cudaMemcpy(NDArray_DATA(out_f16), NDArray_DATA(slice_f16),
                           (size_t)n_per_slice * sizeof(uint16_t),
                           cudaMemcpyDeviceToDevice);
            } else {
                gpu_axis_inplace_op("float16", NDArray_DATA(out_f16),
                                    NDArray_DATA(slice_f16), n_slice_i, op);
            }
        }
        NDArray_FREE(slice_f16);
        if (!strcmp(out_dt, "float4")) {
            cuda_cast_f16_to_fp4((uint16_t *)NDArray_DATA(out_f16),
                                 (uint8_t *)out_data, (int)n_per_slice);
        } else {
            cuda_cast_f16_to_fp8((uint16_t *)NDArray_DATA(out_f16),
                                 (uint8_t *)out_data, (int)n_per_slice);
        }
        NDArray_FREE(out_f16);
        if (a_rolled != a_widened) NDArray_FREE(a_rolled);
        if (a_widened != a) NDArray_FREE(a_widened);
        return out;
    }

    /* Native-kernel dtypes. */
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

/* ──────────────────────────────────────────────────────────────────────────
   Native int64 / uint64 CPU arithmetic kernels. The default arithmetic
   dispatch promotes both `int64` and `uint64` inputs to `float64` for the
   computation; that round-trip through a 53-bit mantissa loses precision
   for any value past 2⁵³ (and silently overflows on inverse cast when the
   intermediate exceeds the dtype's range). The kernels below operate
   directly on the native integer storage so wide values survive end-to-end.

   Identical broadcast / 0-D scalar / shape-mismatch handling as the
   float kernels. `b_temp` / `a_temp` track scalar-broadcast NDArrays so
   the cleanup path can free them exactly once.
   ────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Body of `NDArray_TypedBinOp_CPU_Int64` / `_UInt64` — shared by both
 *        wraps to avoid duplicating the broadcast / alloc plumbing.
 *
 * @param[in] a, b     Caller-validated operands of the same dtype (`int64`
 *                     or `uint64`) on CPU. One operand may be 0-D.
 * @param[in] opcode   ZEND_ADD / SUB / MUL / MOD / POW (DIV routes to float).
 * @param[in] is_signed Non-zero when both operands are `int64` (controls
 *                     mod / pow behaviour for negative bases / negative
 *                     exponents).
 * @return Freshly-allocated NDArray on success (caller owns), NULL on
 *         validation error (PHP exception in flight).
 */
static NDArray *
ndarray_int64_binop_cpu(NDArray *a, NDArray *b, int opcode, int is_signed) {
    NDArray *a_temp = NULL, *b_temp = NULL, *broadcasted = NULL;
    const char *out_dt = NDArray_TYPE(a);

    /* 0-D scalar broadcast — fill a temporary buffer of `other`'s shape with
       the scalar's single element. The scalar may be int64 or uint64; we
       compare dtypes only via the canonical string. */
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        memcpy(n_shape, NDArray_SHAPE(b), sizeof(int) * NDArray_NDIM(b));
        a = NDArray_Empty(n_shape, NDArray_NDIM(b), out_dt, NDARRAY_DEVICE_CPU);
        if (a == NULL) return NULL;
        if (is_signed) {
            int64_t v;
            memcpy(&v, NDArray_DATA(a_temp), sizeof(int64_t));
            int64_t *dst = (int64_t *)NDArray_DATA(a);
            for (long i = 0; i < NDArray_NUMELEMENTS(a); i++) dst[i] = v;
        } else {
            uint64_t v;
            memcpy(&v, NDArray_DATA(a_temp), sizeof(uint64_t));
            uint64_t *dst = (uint64_t *)NDArray_DATA(a);
            for (long i = 0; i < NDArray_NUMELEMENTS(a); i++) dst[i] = v;
        }
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(n_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
        b = NDArray_Empty(n_shape, NDArray_NDIM(a), out_dt, NDARRAY_DEVICE_CPU);
        if (b == NULL) {
            if (a_temp) NDArray_FREE(a);
            return NULL;
        }
        if (is_signed) {
            int64_t v;
            memcpy(&v, NDArray_DATA(b_temp), sizeof(int64_t));
            int64_t *dst = (int64_t *)NDArray_DATA(b);
            for (long i = 0; i < NDArray_NUMELEMENTS(b); i++) dst[i] = v;
        } else {
            uint64_t v;
            memcpy(&v, NDArray_DATA(b_temp), sizeof(uint64_t));
            uint64_t *dst = (uint64_t *)NDArray_DATA(b);
            for (long i = 0; i < NDArray_NUMELEMENTS(b); i++) dst[i] = v;
        }
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
    if (is_signed) {
        int64_t *ap = (int64_t *)NDArray_DATA(a_broad);
        int64_t *bp = (int64_t *)NDArray_DATA(b_broad);
        int64_t *rp = (int64_t *)NDArray_DATA(result);
        switch (opcode) {
            case ZEND_ADD: for (long i = 0; i < n; i++) rp[i] = (int64_t)((uint64_t)ap[i] + (uint64_t)bp[i]); break;
            case ZEND_SUB: for (long i = 0; i < n; i++) rp[i] = (int64_t)((uint64_t)ap[i] - (uint64_t)bp[i]); break;
            case ZEND_MUL: for (long i = 0; i < n; i++) rp[i] = (int64_t)((uint64_t)ap[i] * (uint64_t)bp[i]); break;
            case ZEND_MOD:
                for (long i = 0; i < n; i++) {
                    /* C99 `%` on signed: implementation-defined for negative
                       operands pre-C11. We rely on C11 truncated semantics
                       (result has sign of dividend), which matches PyTorch's
                       `torch.remainder` for ints. Divide-by-zero → 0 to
                       avoid SIGFPE; matches the float kernels' NaN convention
                       in spirit. */
                    rp[i] = (bp[i] == 0) ? 0 : (ap[i] % bp[i]);
                }
                break;
            case ZEND_POW:
                for (long i = 0; i < n; i++) {
                    int64_t base = ap[i], exp = bp[i];
                    if (exp < 0) { rp[i] = 0; continue; }
                    int64_t r = 1;
                    while (exp > 0) {
                        if (exp & 1) r = (int64_t)((uint64_t)r * (uint64_t)base);
                        base = (int64_t)((uint64_t)base * (uint64_t)base);
                        exp >>= 1;
                    }
                    rp[i] = r;
                }
                break;
            default:
                zend_throw_error(NULL, "Unsupported opcode for int64 CPU binop.");
                NDArray_FREE(result);
                if (a_temp) NDArray_FREE(a);
                if (b_temp) NDArray_FREE(b);
                if (broadcasted) NDArray_FREE(broadcasted);
                return NULL;
        }
    } else {
        uint64_t *ap = (uint64_t *)NDArray_DATA(a_broad);
        uint64_t *bp = (uint64_t *)NDArray_DATA(b_broad);
        uint64_t *rp = (uint64_t *)NDArray_DATA(result);
        switch (opcode) {
            case ZEND_ADD: for (long i = 0; i < n; i++) rp[i] = ap[i] + bp[i]; break;
            case ZEND_SUB: for (long i = 0; i < n; i++) rp[i] = ap[i] - bp[i]; break;
            case ZEND_MUL: for (long i = 0; i < n; i++) rp[i] = ap[i] * bp[i]; break;
            case ZEND_MOD:
                for (long i = 0; i < n; i++) {
                    rp[i] = (bp[i] == 0) ? 0 : (ap[i] % bp[i]);
                }
                break;
            case ZEND_POW:
                for (long i = 0; i < n; i++) {
                    uint64_t base = ap[i], exp = bp[i];
                    uint64_t r = 1;
                    while (exp > 0) {
                        if (exp & 1) r *= base;
                        base *= base;
                        exp >>= 1;
                    }
                    rp[i] = r;
                }
                break;
            default:
                zend_throw_error(NULL, "Unsupported opcode for uint64 CPU binop.");
                NDArray_FREE(result);
                if (a_temp) NDArray_FREE(a);
                if (b_temp) NDArray_FREE(b);
                if (broadcasted) NDArray_FREE(broadcasted);
                return NULL;
        }
    }

    if (a_temp) NDArray_FREE(a);
    if (b_temp) NDArray_FREE(b);
    if (broadcasted) NDArray_FREE(broadcasted);
    return result;
}

/**
 * @brief CPU binary-op dispatcher for `int64` / `uint64`.
 *
 * Routes the supported opcodes to the native-int kernel above so wide
 * values stay loss-free. Falls back to NULL with a PHP error for opcodes
 * outside the supported set — the caller (`ndarray_promote_and_op`) only
 * funnels +, -, *, %, ** here; / is already promoted to float by
 * `ndarray_div_promote`.
 *
 * @param[in] opcode ZEND_ADD / SUB / MUL / MOD / POW.
 * @param[in] a, b   Same-dtype operands (`int64` or `uint64`), CPU resident.
 * @return Result NDArray on success, NULL on error.
 */
NDArray *
NDArray_TypedBinOp_CPU_Int64(int opcode, NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) != NDARRAY_DEVICE_CPU
        || NDArray_DEVICE(b) != NDARRAY_DEVICE_CPU) {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_CPU_Int64: both operands must be on CPU.");
        return NULL;
    }
    if (strcmp(NDArray_TYPE(a), NDArray_TYPE(b)) != 0) {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_CPU_Int64: dtype mismatch (%s vs %s).",
            NDArray_TYPE(a), NDArray_TYPE(b));
        return NULL;
    }
    int is_signed;
    if (!strcmp(NDArray_TYPE(a), "int64"))  is_signed = 1;
    else if (!strcmp(NDArray_TYPE(a), "uint64")) is_signed = 0;
    else {
        zend_throw_error(NULL,
            "NDArray_TypedBinOp_CPU_Int64: unsupported dtype \"%s\".",
            NDArray_TYPE(a));
        return NULL;
    }
    return ndarray_int64_binop_cpu(a, b, opcode, is_signed);
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
