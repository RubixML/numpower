#ifndef PHPSCI_NDARRAY_CUDAMATH_H
#define PHPSCI_NDARRAY_CUDAMATH_H

#include "../../ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ElementWiseFloatGPUOperation)(int, float *);
typedef void (*ElementWiseFloatGPUOperation2F)(int, float *, float, float);
typedef void (*ElementWiseFloatGPUOperation1F)(int, float *, float);
typedef void (*ElementWiseFloatGPUOperation1N)(int, float *, float *);
NDArray* NDArrayMathGPU_ElementWise1N(NDArray* ndarray, ElementWiseFloatGPUOperation1N op, NDArray* val1);
NDArray* NDArrayMathGPU_ElementWise(NDArray *ndarray, ElementWiseFloatGPUOperation op);
void cuda_float_abs(int nblocks, float *d_array);
void cuda_float_expm1(int nblocks, float *d_array);
void cuda_float_exp(int nblocks, float *d_array);
void cuda_float_sqrt(int nblocks, float *d_array);
void cuda_float_log(int nblocks, float *d_array);
void cuda_float_logb(int nblocks, float *d_array);
void cuda_float_log2(int nblocks, float *d_array);
void cuda_float_log1p(int nblocks, float *d_array);
void cuda_float_log10(int nblocks, float *d_array);
void cuda_add_float(int nblocks, float *a, float *b, float *rtn, int nelements);
void cuda_subtract_float(int nblocks, float *a, float *b, float *rtn, int nelements);
void cuda_divide_float(int nblocks, float *a, float *b, float *rtn, int nelements);
void cuda_multiply_float(int nblocks, float *a, float *b, float *rtn, int nelements);
void cuda_mod_float(int nblocks, float *a, float *b, float *rtn, int nelements);
int cuda_svd_float(float *d_A, float *d_U, float *d_V, float *d_S, int m, int n);
float cuda_max_float(float *a, int nelements);
float cuda_min_float(float *a, int nelements);
void cuda_pow_float(int nblocks, float *a, float *b, float *rtn, int nelements);
int cuda_equal_float(int nblocks, float *a, float *b, int nelements);
void cuda_sum_float(int nblocks, float *a, float *rtn, int nelements);

void cuda_fill_float(float *a, float value, int n);
void cuda_fill_double(double *a, double value, int n);

int cuda_det_float(float *a, float *result, int n);
void cuda_float_sin(int nblocks, float *d_array);
void cuda_float_cos(int nblocks, float *d_array);
void cuda_float_tan(int nblocks, float *d_array);
void cuda_float_arcsin(int nblocks, float *d_array);
void cuda_float_arccos(int nblocks, float *d_array);
void cuda_float_arctan(int nblocks, float *d_array);
void cuda_float_arctan2(int nblocks, float *d_array, float *y_array);
void cuda_float_degrees(int nblocks, float *d_array);
void cuda_float_radians(int nblocks, float *d_array);
void cuda_float_sinh(int nblocks, float *d_array);
void cuda_float_cosh(int nblocks, float *d_array);
void cuda_float_tanh(int nblocks, float *d_array);
void cuda_float_arcsinh(int nblocks, float *d_array);
void cuda_float_arccosh(int nblocks, float *d_array);
void cuda_float_arctanh(int nblocks, float *d_array);
void cuda_float_rint(int nblocks, float *d_array);
void cuda_float_fix(int nblocks, float *d_array);
void cuda_float_ceil(int nblocks, float *d_array);
void cuda_float_floor(int nblocks, float *d_array);
void cuda_float_sinc(int nblocks, float *d_array);
void cuda_float_trunc(int nblocks, float *d_array);
void cuda_float_negate(int nblocks, float *d_array);
void cuda_float_sign(int nblocks, float *d_array);
void cuda_float_clip(int nblocks, float *d_array, float minVal, float maxVal);
void cuda_float_multiply_matrix_vector(int nblocks, float *a_array, float *b_array, float *result, int rows, int cols);
void cuda_float_compare_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
void cuda_matrix_float_inverse(float* matrix, int n);
void cuda_float_lu(float *matrix, float *L, float *U, float *P, int size);
void cuda_prod_float(int nblocks, float *a, float *rtn, int nelements);
void cuda_float_round(int nblocks, float *d_array, float decimals);
void cuda_calculate_outer_product(int m, int n, float *a_array, float *b_array, float *r_array);
void cuda_float_compare_greater(int nblocks, float *a_array, float *b_array, float *result, int n);
void cuda_float_compare_greater_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
void cuda_float_compare_less(int nblocks, float *a_array, float *b_array, float *result, int n);
void cuda_float_compare_less_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
void cuda_float_compare_not_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
/* cuda_lstsq_float was historically declared here but never implemented in
   cuda_math.cu. The call site in src/ndmath/linalg.c (NDArray_Lstsq, GPU
   branch) now throws "not implemented for GPU" instead of referencing a
   non-existent symbol. Re-add the declaration here when the GPU kernel
   actually lands. */
NDArray* NDArrayMathGPU_ElementWise2F(NDArray* ndarray, ElementWiseFloatGPUOperation2F op, float val1, float val2);
NDArray* NDArrayMathGPU_ElementWise1F(NDArray* ndarray, ElementWiseFloatGPUOperation1F op, float val1);
void cuda_float_transpose(int tiledim, int blockrows, const float *d_in, float *d_out, int width, int height);
void cuda_float_positive(int nblocks, float *d_array);
void cuda_float_reciprocal(int nblocks, float *d_array);
/**
 * @brief Fill a GPU float32 buffer with truncated-Gaussian samples.
 *
 * Per-element value is `loc + scale * z` where `z ~ N(0, 1)` is
 * rejection-sampled to lie in `[-2, 2]`. Stored values are therefore in
 * `[loc - 2σ, loc + 2σ]`. Each thread runs its own cuRAND state seeded
 * from `(seed, idx)` so the streams are independent; the seed itself is
 * fresh per call via `cuda_normal_next_seed` (was pinned to `1234ULL`
 * in the previous implementation, producing identical samples on every
 * call within one process).
 *
 * @param[out] d_data Destination GPU buffer of @p n floats.
 * @param[in]  n      Element count; ≥ 0.
 * @param[in]  loc    Distribution mean (µ).
 * @param[in]  scale  Distribution standard deviation (σ); must be > 0.
 */
void cuda_truncated_normal_f32(float *d_data, long n, float loc, float scale);

/**
 * @brief Fill a GPU float64 buffer with truncated-Gaussian samples.
 *
 * Companion to `cuda_truncated_normal_f32` for double precision. The
 * underlying samples carry 53-bit precision before the affine
 * transform; truncation bounds are evaluated in double too.
 *
 * @param[out] d_data Destination GPU buffer of @p n doubles.
 * @param[in]  n      Element count; ≥ 0.
 * @param[in]  loc    Distribution mean (µ).
 * @param[in]  scale  Distribution standard deviation (σ); must be > 0.
 */
void cuda_truncated_normal_f64(double *d_data, long n, double loc, double scale);

/**
 * @brief Fill a GPU float32 buffer with N(@p mean, @p stddev^2) samples.
 *
 * Wraps `curandGenerateNormal`. cuRAND's contract requires an even sample
 * count, so the wrapper transparently uses an internal +1 padding buffer
 * when @p n is odd. The seed is derived from `time(NULL)` xored with a
 * monotonically-increasing call counter so successive calls inside the
 * same second still produce independent streams.
 *
 * @param[out] d_data  GPU buffer of @p n floats.
 * @param[in]  n       Element count; ≥ 0.
 * @param[in]  mean    Distribution mean (µ).
 * @param[in]  stddev  Distribution standard deviation (σ); must be > 0
 *                     for cuRAND to be well-defined.
 */
void cuda_normal_f32(float *d_data, long n, float mean, float stddev);

/**
 * @brief Fill a GPU float64 buffer with N(@p mean, @p stddev^2) samples.
 *
 * Companion to `cuda_normal_f32` for double precision. Same odd-size
 * handling and same per-call seeding policy.
 *
 * @param[out] d_data  GPU buffer of @p n doubles.
 * @param[in]  n       Element count; ≥ 0.
 * @param[in]  mean    Distribution mean (µ).
 * @param[in]  stddev  Distribution standard deviation (σ); must be > 0.
 */
void cuda_normal_f64(double *d_data, long n, double mean, double stddev);

/**
 * @brief Widen a GPU float64 normal stream into the on-device DD layout.
 *
 * Reads @p z[i] (a standard-normal double), evaluates the affine
 * transform `value = loc + scale * z[i]` in true double-double
 * arithmetic on device using the same `dd_add` / `dd_mul` primitives the
 * arithmetic kernels use, and stores the result at
 * `dst[2*i]` (hi) and `dst[2*i + 1]` (lo). Used by `NDArray_Normal` to
 * keep the fp128 GPU path VRAM-direct (no host transit of the result).
 *
 * @param[in]  z         Length-@p n GPU buffer of standard-normal doubles.
 * @param[out] dst       Length-`2*n` GPU buffer of interleaved (hi, lo) pairs.
 * @param[in]  n         Element count.
 * @param[in]  loc_hi    DD high word of the distribution mean.
 * @param[in]  loc_lo    DD low word of the distribution mean.
 * @param[in]  scale_hi  DD high word of the distribution stddev.
 * @param[in]  scale_lo  DD low word of the distribution stddev.
 */
void cuda_normal_dd_affine(const double *z, double *dst, long n,
                           double loc_hi, double loc_lo,
                           double scale_hi, double scale_lo);

//Doubles
void cuda_sum_double(int nblocks, double *a, double *rtn, int nelements);

#include <stdint.h>

/* Typed binary ops — each operates on a GPU buffer of length n. */
void cuda_add_i8 (int8_t  *a, int8_t  *b, int8_t  *rtn, int n);
void cuda_sub_i8 (int8_t  *a, int8_t  *b, int8_t  *rtn, int n);
void cuda_mul_i8 (int8_t  *a, int8_t  *b, int8_t  *rtn, int n);
void cuda_div_i8 (int8_t  *a, int8_t  *b, int8_t  *rtn, int n);
void cuda_mod_i8 (int8_t  *a, int8_t  *b, int8_t  *rtn, int n);
void cuda_pow_i8 (int8_t  *a, int8_t  *b, int8_t  *rtn, int n);
void cuda_add_u8 (uint8_t *a, uint8_t *b, uint8_t *rtn, int n);
void cuda_sub_u8 (uint8_t *a, uint8_t *b, uint8_t *rtn, int n);
void cuda_mul_u8 (uint8_t *a, uint8_t *b, uint8_t *rtn, int n);
void cuda_div_u8 (uint8_t *a, uint8_t *b, uint8_t *rtn, int n);
void cuda_mod_u8 (uint8_t *a, uint8_t *b, uint8_t *rtn, int n);
void cuda_pow_u8 (uint8_t *a, uint8_t *b, uint8_t *rtn, int n);
void cuda_add_i16(int16_t  *a, int16_t  *b, int16_t  *rtn, int n);
void cuda_sub_i16(int16_t  *a, int16_t  *b, int16_t  *rtn, int n);
void cuda_mul_i16(int16_t  *a, int16_t  *b, int16_t  *rtn, int n);
void cuda_div_i16(int16_t  *a, int16_t  *b, int16_t  *rtn, int n);
void cuda_mod_i16(int16_t  *a, int16_t  *b, int16_t  *rtn, int n);
void cuda_pow_i16(int16_t  *a, int16_t  *b, int16_t  *rtn, int n);
void cuda_add_u16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_sub_u16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_mul_u16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_div_u16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_mod_u16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_pow_u16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_add_i32(int32_t  *a, int32_t  *b, int32_t  *rtn, int n);
void cuda_sub_i32(int32_t  *a, int32_t  *b, int32_t  *rtn, int n);
void cuda_mul_i32(int32_t  *a, int32_t  *b, int32_t  *rtn, int n);
void cuda_div_i32(int32_t  *a, int32_t  *b, int32_t  *rtn, int n);
void cuda_mod_i32(int32_t  *a, int32_t  *b, int32_t  *rtn, int n);
void cuda_pow_i32(int32_t  *a, int32_t  *b, int32_t  *rtn, int n);
void cuda_add_u32(uint32_t *a, uint32_t *b, uint32_t *rtn, int n);
void cuda_sub_u32(uint32_t *a, uint32_t *b, uint32_t *rtn, int n);
void cuda_mul_u32(uint32_t *a, uint32_t *b, uint32_t *rtn, int n);
void cuda_div_u32(uint32_t *a, uint32_t *b, uint32_t *rtn, int n);
void cuda_mod_u32(uint32_t *a, uint32_t *b, uint32_t *rtn, int n);
void cuda_pow_u32(uint32_t *a, uint32_t *b, uint32_t *rtn, int n);
void cuda_add_i64(int64_t  *a, int64_t  *b, int64_t  *rtn, int n);
void cuda_sub_i64(int64_t  *a, int64_t  *b, int64_t  *rtn, int n);
void cuda_mul_i64(int64_t  *a, int64_t  *b, int64_t  *rtn, int n);
void cuda_div_i64(int64_t  *a, int64_t  *b, int64_t  *rtn, int n);
void cuda_mod_i64(int64_t  *a, int64_t  *b, int64_t  *rtn, int n);
void cuda_pow_i64(int64_t  *a, int64_t  *b, int64_t  *rtn, int n);
void cuda_add_u64(uint64_t *a, uint64_t *b, uint64_t *rtn, int n);
void cuda_sub_u64(uint64_t *a, uint64_t *b, uint64_t *rtn, int n);
void cuda_mul_u64(uint64_t *a, uint64_t *b, uint64_t *rtn, int n);
void cuda_div_u64(uint64_t *a, uint64_t *b, uint64_t *rtn, int n);
void cuda_mod_u64(uint64_t *a, uint64_t *b, uint64_t *rtn, int n);
void cuda_pow_u64(uint64_t *a, uint64_t *b, uint64_t *rtn, int n);

/* float16 buffers are stored as uint16_t since <cuda_fp16.h> is CUDA-only. */
void cuda_add_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_sub_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_mul_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_div_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_mod_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);
void cuda_pow_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n);

void cuda_add_f64(double *a, double *b, double *rtn, int n);
void cuda_sub_f64(double *a, double *b, double *rtn, int n);
void cuda_mul_f64(double *a, double *b, double *rtn, int n);
void cuda_div_f64(double *a, double *b, double *rtn, int n);
void cuda_mod_f64(double *a, double *b, double *rtn, int n);
void cuda_pow_f64(double *a, double *b, double *rtn, int n);

/* Typed fills */
void cuda_fill_i8 (int8_t   *a, int8_t   value, int n);
void cuda_fill_u8 (uint8_t  *a, uint8_t  value, int n);
void cuda_fill_i16(int16_t  *a, int16_t  value, int n);
void cuda_fill_u16(uint16_t *a, uint16_t value, int n);
void cuda_fill_i32(int32_t  *a, int32_t  value, int n);
void cuda_fill_u32(uint32_t *a, uint32_t value, int n);
void cuda_fill_i64(int64_t  *a, int64_t  value, int n);
void cuda_fill_u64(uint64_t *a, uint64_t value, int n);
void cuda_fill_f16(uint16_t *a, float    value, int n);
void cuda_fill_f64(double   *a, double   value, int n);
void cuda_fill_dd (double   *out, double hi, double lo, int n);

/* Generic byte-wise broadcast: dst[i] = src[src_offsets[i]]. src_offsets_gpu
   is a GPU buffer of n_out ints precomputed host-side per broadcast rule. */
void cuda_broadcast(const char *src_gpu, char *dst_gpu,
                    const int *src_offsets_gpu, int n_out, int elsize);

/* float4 / float8 casts on GPU. fp4 / fp8 are 1 byte per element. */
void cuda_cast_fp4_to_f32(uint8_t *src, float *dst, int n);
void cuda_cast_f32_to_fp4(float *src, uint8_t *dst, int n);
void cuda_cast_fp4_to_f16(uint8_t *src, uint16_t *dst, int n);
void cuda_cast_f16_to_fp4(uint16_t *src, uint8_t *dst, int n);
void cuda_cast_fp8_to_f32(uint8_t *src, float *dst, int n);
void cuda_cast_f32_to_fp8(float *src, uint8_t *dst, int n);
void cuda_cast_fp8_to_f16(uint8_t *src, uint16_t *dst, int n);
void cuda_cast_f16_to_fp8(uint16_t *src, uint8_t *dst, int n);

/* Double-double (GPU emulation of float128) — operates on a buffer of length
   2*n doubles laid out as (hi[0], lo[0], hi[1], lo[1], ...). */
void cuda_add_dd(double *a, double *b, double *rtn, int n);
void cuda_sub_dd(double *a, double *b, double *rtn, int n);
void cuda_mul_dd(double *a, double *b, double *rtn, int n);
void cuda_div_dd(double *a, double *b, double *rtn, int n);
void cuda_pow_dd(double *a, double *b, double *rtn, int n);
void cuda_mod_dd(double *a, double *b, double *rtn, int n);

/* GPU AsType cast wrappers. Naming: cuda_cast_<src>_to_<dst> */
#define CUDA_CAST_DECL(SRC, DST, ST, DT) \
    void cuda_cast_##SRC##_to_##DST(ST *src, DT *dst, int n);

#define CUDA_CAST_DECL_ALL_DST(SRC, ST) \
    CUDA_CAST_DECL(SRC, i8,  ST, int8_t)   \
    CUDA_CAST_DECL(SRC, u8,  ST, uint8_t)  \
    CUDA_CAST_DECL(SRC, i16, ST, int16_t)  \
    CUDA_CAST_DECL(SRC, u16, ST, uint16_t) \
    CUDA_CAST_DECL(SRC, i32, ST, int32_t)  \
    CUDA_CAST_DECL(SRC, u32, ST, uint32_t) \
    CUDA_CAST_DECL(SRC, i64, ST, int64_t)  \
    CUDA_CAST_DECL(SRC, u64, ST, uint64_t) \
    CUDA_CAST_DECL(SRC, f32, ST, float)    \
    CUDA_CAST_DECL(SRC, f64, ST, double)

CUDA_CAST_DECL_ALL_DST(i8,  int8_t)
CUDA_CAST_DECL_ALL_DST(u8,  uint8_t)
CUDA_CAST_DECL_ALL_DST(i16, int16_t)
CUDA_CAST_DECL_ALL_DST(u16, uint16_t)
CUDA_CAST_DECL_ALL_DST(i32, int32_t)
CUDA_CAST_DECL_ALL_DST(u32, uint32_t)
CUDA_CAST_DECL_ALL_DST(i64, int64_t)
CUDA_CAST_DECL_ALL_DST(u64, uint64_t)
CUDA_CAST_DECL_ALL_DST(f32, float)
CUDA_CAST_DECL_ALL_DST(f64, double)

/* half ⇄ everything */
void cuda_cast_f16_to_i8 (uint16_t *src, int8_t   *dst, int n);
void cuda_cast_f16_to_u8 (uint16_t *src, uint8_t  *dst, int n);
void cuda_cast_f16_to_i16(uint16_t *src, int16_t  *dst, int n);
void cuda_cast_f16_to_u16(uint16_t *src, uint16_t *dst, int n);
void cuda_cast_f16_to_i32(uint16_t *src, int32_t  *dst, int n);
void cuda_cast_f16_to_u32(uint16_t *src, uint32_t *dst, int n);
void cuda_cast_f16_to_i64(uint16_t *src, int64_t  *dst, int n);
void cuda_cast_f16_to_u64(uint16_t *src, uint64_t *dst, int n);
void cuda_cast_f16_to_f32(uint16_t *src, float    *dst, int n);
void cuda_cast_f16_to_f64(uint16_t *src, double   *dst, int n);
void cuda_cast_f16_to_f16(uint16_t *src, uint16_t *dst, int n);
void cuda_cast_i8_to_f16 (int8_t   *src, uint16_t *dst, int n);
void cuda_cast_u8_to_f16 (uint8_t  *src, uint16_t *dst, int n);
void cuda_cast_i16_to_f16(int16_t  *src, uint16_t *dst, int n);
void cuda_cast_u16_to_f16(uint16_t *src, uint16_t *dst, int n);
void cuda_cast_i32_to_f16(int32_t  *src, uint16_t *dst, int n);
void cuda_cast_u32_to_f16(uint32_t *src, uint16_t *dst, int n);
void cuda_cast_i64_to_f16(int64_t  *src, uint16_t *dst, int n);
void cuda_cast_u64_to_f16(uint64_t *src, uint16_t *dst, int n);
void cuda_cast_f32_to_f16(float    *src, uint16_t *dst, int n);
void cuda_cast_f64_to_f16(double   *src, uint16_t *dst, int n);

/**
 * Strided device-to-device copy.
 *
 * Materialises a contiguous result NDArray from a view described by
 * (ndim, shape, strides_b) in a single kernel launch — used by
 * NDArray_Slice on GPU to avoid the per-row cudaMemcpy latency that
 * dominates strided copies. Strides are in BYTES and may be negative
 * (for reverse-step slices).
 *
 * @param dst_gpu          Pre-allocated device destination buffer of
 *                         `n_elems * elsize` bytes; written contiguously.
 * @param src_gpu          Device source buffer; read via `host_strides_b`.
 * @param n_elems          Total output element count (= product of `host_shape`).
 * @param ndim             Number of axes; must be in [1, STRIDED_COPY_MAX_NDIM].
 * @param elsize           Dtype byte size; fast paths cover 1/2/4/8/16.
 * @param host_shape       Per-axis shape (host memory, packed into kernel arg).
 * @param host_strides_b   Per-axis source stride in bytes, signed.
 * @return                 0 on success, -1 if (ndim, elsize) are outside the
 *                         kernel envelope so the caller can pick a fallback.
 */
int cuda_strided_copy(char *dst_gpu, const char *src_gpu,
                      long long n_elems, int ndim, int elsize,
                      const int       *host_shape,
                      const long long *host_strides_b);

/* ── Typed reductions ───────────────────────────────────────────────────────
 * Each function consumes a GPU-resident source buffer of @p n elements and
 * an already-seeded one-double accumulator on GPU. The caller must:
 *   1. vmalloc a double slot on GPU.
 *   2. cudaMemcpy the identity value into that slot
 *      (0 for sum, 1 for prod, +DBL_MAX for min, -DBL_MAX for max).
 *   3. Launch the kernel via the wrapper below.
 *   4. cudaMemcpy the result back to host and vfree the slot.
 *
 * The kernels are single-pass atomic-only — no shared memory, no warp
 * shuffles — so they work for any block / grid configuration without UB.
 * Performance trade-off: atomic contention for very large n is bounded
 * by the device's L2 / atomic throughput, which is more than sufficient
 * for typical PHP-driven workloads.
 *
 * Dtype tags map 1:1 to the NDARRAY_TYPE_* strings:
 *   i8..i64, u8..u64 = native integers (atomicAdd-castable to double)
 *   f16   = __half  (widened via __half2float in the kernel)
 *   f32   = float
 *   f64   = double
 *   fp4   = 4-bit float in lower nibble of uint8 (decoded via dev_fp4_to_float)
 *   fp8   = E4M3 in uint8 (decoded via dev_fp8_to_float)
 *   dd    = double-double interleaved (hi, lo) — GPU representation of fp128
 */
#define DECL_REDUCE_TYPED(OP, T, TAG)                                                \
    void cuda_reduce_##OP##_##TAG(const T *a, double *result, int n);

#define DECL_REDUCE_BUNDLE(OP)                                                       \
    DECL_REDUCE_TYPED(OP, int8_t,   i8)                                              \
    DECL_REDUCE_TYPED(OP, uint8_t,  u8)                                              \
    DECL_REDUCE_TYPED(OP, int16_t,  i16)                                             \
    DECL_REDUCE_TYPED(OP, uint16_t, u16)                                             \
    DECL_REDUCE_TYPED(OP, int32_t,  i32)                                             \
    DECL_REDUCE_TYPED(OP, uint32_t, u32)                                             \
    DECL_REDUCE_TYPED(OP, int64_t,  i64)                                             \
    DECL_REDUCE_TYPED(OP, uint64_t, u64)                                             \
    DECL_REDUCE_TYPED(OP, float,    f32)                                             \
    DECL_REDUCE_TYPED(OP, double,   f64)                                             \
    DECL_REDUCE_TYPED(OP, uint16_t, f16)                                             \
    DECL_REDUCE_TYPED(OP, uint8_t,  fp4)                                             \
    DECL_REDUCE_TYPED(OP, uint8_t,  fp8)                                             \
    DECL_REDUCE_TYPED(OP, double,   dd)

DECL_REDUCE_BUNDLE(sum)
DECL_REDUCE_BUNDLE(prod)
DECL_REDUCE_BUNDLE(min)
DECL_REDUCE_BUNDLE(max)

#undef DECL_REDUCE_BUNDLE
#undef DECL_REDUCE_TYPED

#ifdef __cplusplus
}
#endif
#endif //PHPSCI_NDARRAY_CUDAMATH_H