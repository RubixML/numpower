#ifndef PHPSCI_NDARRAY_CUDAMATH_H
#define PHPSCI_NDARRAY_CUDAMATH_H

#include "../../ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ElementWiseFloatGPUOperation)(int, float *);
typedef void (*ElementWiseFloatGPUOperation2F)(int, float *, float, float);
typedef void (*ElementWiseFloatGPUOperation1N)(int, float *, float *);
NDArray* NDArrayMathGPU_ElementWise1N(NDArray* ndarray, ElementWiseFloatGPUOperation1N op, NDArray* val1);
NDArray* NDArrayMathGPU_ElementWise(NDArray *ndarray, ElementWiseFloatGPUOperation op);
void cuda_float_abs(int nblocks, float *d_array);
void cuda_float_sqrt(int nblocks, float *d_array);
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
/* Legacy single-precision trig / hyperbolic / angle / rounding /
   sinc / negate / sign / clip helpers were removed from the public
   API by the typed-unary refactor — all dispatch goes through
   `cuda_<op>_{f32,f64,f16,dd}` now. `arctan2` joined the typed binary
   dispatch (`cuda_atan2_{f32,f64,dd}`, declared with the other typed
   binops below) and `round` joined the typed unary dispatch
   (`cuda_round_{f16,f32,f64,dd}`, declared below). No legacy float-only
   element-wise wrapper remains. */
void cuda_float_multiply_matrix_vector(int nblocks, float *a_array, float *b_array, float *result, int rows, int cols);
void cuda_float_compare_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
void cuda_matrix_float_inverse(float* matrix, int n);
void cuda_float_lu(float *matrix, float *L, float *U, float *P, int size);
void cuda_prod_float(int nblocks, float *a, float *rtn, int nelements);
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
void cuda_float_transpose(int tiledim, int blockrows, const float *d_in, float *d_out, int width, int height);
/* `cuda_float_positive` / `cuda_float_reciprocal` removed from the
   public API by the typed-unary refactor — use cuda_recip_{f16,f32,
   f64,dd} (reciprocal) or omit (positive is a no-op for floats and
   handled by the dispatcher). */
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

/**
 * @brief Fill a GPU float32 buffer with uniform `[low, high)` samples.
 *
 * Wraps `curandGenerateUniform` to produce `(0, 1]` samples directly
 * into @p d_data, then reflects via `1 - u` and applies the affine
 * `low + (1 - u) * (high - low)` so the closed endpoint sits at
 * @p low (matching numpy's `[low, high)` contract). cuRAND's uniform
 * generator has no parity restriction, so the call is single-pass into
 * the destination with no pad buffer. The seed is fresh per call
 * (`cuda_normal_next_seed`) so successive calls in the same process
 * produce independent streams.
 *
 * @param[out] d_data Destination GPU buffer of @p n floats.
 * @param[in]  n      Element count; ≥ 0.
 * @param[in]  low    Lower bound (inclusive).
 * @param[in]  high   Upper bound (exclusive).
 */
void cuda_uniform_f32(float *d_data, long n, float low, float high);

/**
 * @brief Fill a GPU float64 buffer with uniform `[low, high)` samples.
 *
 * Companion to `cuda_uniform_f32` for double precision. Uses
 * `curandGenerateUniformDouble`; the affine reflection runs in fp64 so
 * the full 53-bit mantissa range is preserved across `[low, high)`.
 *
 * @param[out] d_data Destination GPU buffer of @p n doubles.
 * @param[in]  n      Element count; ≥ 0.
 * @param[in]  low    Lower bound (inclusive).
 * @param[in]  high   Upper bound (exclusive).
 */
void cuda_uniform_f64(double *d_data, long n, double low, double high);

/**
 * @brief Widen a GPU float64 uniform stream into the on-device DD layout.
 *
 * Reads `[0, 1)` doubles from @p u (callers pre-reflect via
 * `cuda_uniform_f64(u, n, 0.0, 1.0)`) and evaluates
 * `value = low + u * range` in true double-double arithmetic on
 * device. `range = high - low` is computed on the host in fp128/DD
 * precision and passed as a DD pair so the caller's fp128 bounds
 * survive bit-for-bit into the interleaved (hi, lo) output. Used by
 * `NDArray_Uniform` to keep the fp128 GPU path VRAM-direct.
 *
 * @param[in]  u         Length-@p n GPU buffer of `[0, 1)` doubles.
 * @param[out] dst       Length-`2*n` GPU buffer of interleaved (hi, lo) pairs.
 * @param[in]  n         Element count.
 * @param[in]  low_hi    DD high word of the lower bound.
 * @param[in]  low_lo    DD low word of the lower bound.
 * @param[in]  range_hi  DD high word of `(high - low)`.
 * @param[in]  range_lo  DD low word of `(high - low)`.
 */
void cuda_uniform_dd_affine(const double *u, double *dst, long n,
                             double low_hi, double low_lo,
                             double range_hi, double range_lo);

/**
 * @brief Apply the normal/truncated-normal `loc + (int64)(scale * z)`
 *        affine on the GPU and store the result as uint64.
 *
 * Reads (possibly truncated) standard-normal doubles from @p z and
 * writes `dst[i] = loc + (uint64_t)((int64_t)(scaled * z[i]))`. The
 * signed→unsigned wrap is intentional — matches numpy's `astype(uint64)`
 * semantics for negative floats and lets a negative z subtract from
 * @p loc through modular arithmetic. Used by `NDArray_Normal` and
 * `NDArray_TruncatedNormal` to keep the u64 GPU path VRAM-direct (no
 * host staging of the result).
 *
 * @param[in]  z      Length-@p n GPU buffer of standard-normal or
 *                    truncated-standard-normal doubles (caller picks
 *                    the source by which cuRAND wrapper populated it).
 * @param[out] dst    Length-@p n GPU uint64 buffer.
 * @param[in]  n      Element count.
 * @param[in]  loc    Distribution mean (uint64).
 * @param[in]  scaled Distribution stddev coerced to double.
 */
void cuda_normal_u64_affine(const double *z, unsigned long long *dst, long n,
                             unsigned long long loc, double scaled);

/**
 * @brief Apply the uniform `low + (uint64)(width * u)` affine on the GPU.
 *
 * Reads `[0, 1)` doubles from @p u (callers pre-reflect via
 * `cuda_uniform_f64(u, n, 0.0, 1.0)`) and writes
 * `dst[i] = low + (uint64_t)(widthd * u[i])`. The width is a `double`
 * because the cast `(double)(high - low)` happens once on the host —
 * widths past 2^53 lose the same precision the CPU filler does
 * (documented invariant). The unsigned add wraps modulo 2^64. Used by
 * `NDArray_Uniform` to keep the u64 GPU path VRAM-direct.
 *
 * @param[in]  u      Length-@p n GPU buffer of `[0, 1)` doubles.
 * @param[out] dst    Length-@p n GPU uint64 buffer.
 * @param[in]  n      Element count.
 * @param[in]  low    Lower bound (uint64).
 * @param[in]  widthd `(double)(high - low)`.
 */
void cuda_uniform_u64_affine(const double *u, unsigned long long *dst, long n,
                              unsigned long long low, double widthd);

/**
 * @brief Fill a GPU uint32 buffer with Poisson(@p lam) samples.
 *
 * Wraps `curandGeneratePoisson`. cuRAND picks an internal algorithm
 * (rejection-from-normal or PTRS) based on @p lam. The library has an
 * undocumented internal precision limit (empirically ~4 × 10^5 on the
 * default XORWOW generator) beyond which the call returns a non-success
 * status; we propagate that as a boolean so callers can raise a clear
 * PHP error instead of returning a silently-zero buffer.
 *
 * There is no parity restriction (unlike the Normal generator), so the
 * call is single-pass into the destination with no pad buffer. The
 * seed is fresh per call (`cuda_normal_next_seed`) so successive calls
 * in the same process produce independent streams.
 *
 * @param[out] d_data Destination GPU buffer of @p n unsigned ints.
 * @param[in]  n      Element count; ≥ 0.
 * @param[in]  lam    Distribution rate (λ); must be ≥ 0.
 * @return 1 on success, 0 if cuRAND rejected the request (typically
 *         due to @p lam exceeding the generator's internal precision
 *         bound; the destination buffer is left untouched).
 */
int cuda_poisson_u32(unsigned int *d_data, long n, double lam);

/**
 * @brief Widen a GPU uint32 stream into the on-device DD layout.
 *
 * Used by `NDArray_Poisson` for the float128 GPU path: each Poisson
 * sample (a non-negative integer up to ~10^9) is written as a DD pair
 * with `hi = (double)src[i]` and `lo = 0.0`. uint32 ≤ 2^32 fits exactly
 * in fp64's 53-bit mantissa, so the high word is byte-correct and the
 * low word carries no residue. Used to keep the fp128 Poisson GPU
 * path VRAM-direct — no host transit of the result.
 *
 * @param[in]  src Length-@p n GPU buffer of uint32 Poisson samples.
 * @param[out] dst Length-`2*n` GPU buffer of interleaved (hi, lo) pairs.
 * @param[in]  n   Element count.
 */
void cuda_cast_u32_to_dd(const unsigned int *src, double *dst, long n);

/**
 * @brief Fill a GPU uint32 buffer with Binomial(@p n, @p p) samples.
 *
 * Direct Bernoulli method: each output slot owns its own per-thread
 * cuRAND state seeded from `(seed, idx)`, runs @p n trials with
 * success probability @p p, and writes the success count as a uint32.
 * Cost is `O(n)` per output element — practical for `n` up to a few
 * thousand. For very large `n` a BTPE-style algorithm would scale
 * better, but the direct method stays numerically exact (the same
 * algorithm as the CPU path), so the distribution contract holds for
 * any `n`.
 *
 * The seed is fresh per call (`cuda_normal_next_seed`) so successive
 * calls in the same process produce statistically-independent streams.
 *
 * @param[out] d_data Destination GPU buffer of @p total unsigned ints.
 * @param[in]  total  Element count; ≥ 0.
 * @param[in]  n      Number of Bernoulli trials per sample.
 * @param[in]  p      Per-trial success probability in `[0, 1]`.
 */
void cuda_binomial_u32(unsigned int *d_data, long total, int n, float p);

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

/* Typed two-argument arctangent (atan2). Only the float compute dtypes are
   needed: the host dispatcher promotes every integer / narrow-float input to
   float32 or float64 first, and routes float128 to `cuda_atan2_dd`. */
void cuda_atan2_f32(float  *a, float  *b, float  *rtn, int n);
void cuda_atan2_f64(double *a, double *b, double *rtn, int n);

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
void cuda_atan2_dd(double *a, double *b, double *rtn, int n);

/* ── Typed unary in-place kernels ───────────────────────────────────────────
   Element-wise unary ops parameterised by dtype. Each wrapper runs the
   computation in place on a device buffer of @p n elements (`2*n` doubles
   for the `_dd` flavours, which interleave (hi, lo)). The dispatcher
   (`NDArray_TypedUnaryOp`) prepares the buffer via `NDArray_Copy` or
   `NDArray_AsType` before launch.

   Sign / abs / square / negate / clip preserve the input dtype. Sqrt,
   rsqrt, reciprocal and sinc only have float instantiations — the
   dispatcher promotes integer inputs to `float32`/`float64` upstream. */
void cuda_negate_i8 (int8_t   *a, int n);
void cuda_negate_u8 (uint8_t  *a, int n);
void cuda_negate_i16(int16_t  *a, int n);
void cuda_negate_u16(uint16_t *a, int n);
void cuda_negate_i32(int32_t  *a, int n);
void cuda_negate_u32(uint32_t *a, int n);
void cuda_negate_i64(int64_t  *a, int n);
void cuda_negate_u64(uint64_t *a, int n);
void cuda_negate_f16(uint16_t *a, int n);
void cuda_negate_f32(float    *a, int n);
void cuda_negate_f64(double   *a, int n);
void cuda_negate_dd (double   *a, int n);

void cuda_abs_i8 (int8_t  *a, int n);
void cuda_abs_i16(int16_t *a, int n);
void cuda_abs_i32(int32_t *a, int n);
void cuda_abs_i64(int64_t *a, int n);
void cuda_abs_f16(uint16_t *a, int n);
void cuda_abs_f32(float   *a, int n);
void cuda_abs_f64(double  *a, int n);
void cuda_abs_dd (double  *a, int n);

void cuda_sign_i8 (int8_t   *a, int n);
void cuda_sign_u8 (uint8_t  *a, int n);
void cuda_sign_i16(int16_t  *a, int n);
void cuda_sign_u16(uint16_t *a, int n);
void cuda_sign_i32(int32_t  *a, int n);
void cuda_sign_u32(uint32_t *a, int n);
void cuda_sign_i64(int64_t  *a, int n);
void cuda_sign_u64(uint64_t *a, int n);
void cuda_sign_f16(uint16_t *a, int n);
void cuda_sign_f32(float    *a, int n);
void cuda_sign_f64(double   *a, int n);
void cuda_sign_dd (double   *a, int n);

void cuda_square_i8 (int8_t   *a, int n);
void cuda_square_u8 (uint8_t  *a, int n);
void cuda_square_i16(int16_t  *a, int n);
void cuda_square_u16(uint16_t *a, int n);
void cuda_square_i32(int32_t  *a, int n);
void cuda_square_u32(uint32_t *a, int n);
void cuda_square_i64(int64_t  *a, int n);
void cuda_square_u64(uint64_t *a, int n);
void cuda_square_f16(uint16_t *a, int n);
void cuda_square_f32(float    *a, int n);
void cuda_square_f64(double   *a, int n);
void cuda_square_dd (double   *a, int n);

void cuda_recip_f16(uint16_t *a, int n);
void cuda_recip_f32(float    *a, int n);
void cuda_recip_f64(double   *a, int n);
void cuda_recip_dd (double   *a, int n);

void cuda_sqrt_f16(uint16_t *a, int n);
void cuda_sqrt_f32(float    *a, int n);
void cuda_sqrt_f64(double   *a, int n);
void cuda_sqrt_dd (double   *a, int n);

void cuda_rsqrt_f16(uint16_t *a, int n);
void cuda_rsqrt_f32(float    *a, int n);
void cuda_rsqrt_f64(double   *a, int n);
void cuda_rsqrt_dd (double   *a, int n);

void cuda_sinc_f16(uint16_t *a, int n);
void cuda_sinc_f32(float    *a, int n);
void cuda_sinc_f64(double   *a, int n);
void cuda_sinc_dd (double   *a, int n);

/* Transcendental wrappers — exp/exp2/expm1/log/log1p/log2/log10/logb.
   Buffer is updated in place; dispatch lives in
   `unary_run_gpu_inplace`. Integer dtypes promote to float32/float64
   upstream so these signatures cover every supported compute dtype. */
void cuda_exp_f16(uint16_t *a, int n);
void cuda_exp_f32(float    *a, int n);
void cuda_exp_f64(double   *a, int n);
void cuda_exp_dd (double   *a, int n);

void cuda_exp2_f16(uint16_t *a, int n);
void cuda_exp2_f32(float    *a, int n);
void cuda_exp2_f64(double   *a, int n);
void cuda_exp2_dd (double   *a, int n);

void cuda_expm1_f16(uint16_t *a, int n);
void cuda_expm1_f32(float    *a, int n);
void cuda_expm1_f64(double   *a, int n);
void cuda_expm1_dd (double   *a, int n);

void cuda_log_f16(uint16_t *a, int n);
void cuda_log_f32(float    *a, int n);
void cuda_log_f64(double   *a, int n);
void cuda_log_dd (double   *a, int n);

void cuda_log1p_f16(uint16_t *a, int n);
void cuda_log1p_f32(float    *a, int n);
void cuda_log1p_f64(double   *a, int n);
void cuda_log1p_dd (double   *a, int n);

void cuda_log2_f16(uint16_t *a, int n);
void cuda_log2_f32(float    *a, int n);
void cuda_log2_f64(double   *a, int n);
void cuda_log2_dd (double   *a, int n);

void cuda_log10_f16(uint16_t *a, int n);
void cuda_log10_f32(float    *a, int n);
void cuda_log10_f64(double   *a, int n);
void cuda_log10_dd (double   *a, int n);

void cuda_logb_f16(uint16_t *a, int n);
void cuda_logb_f32(float    *a, int n);
void cuda_logb_f64(double   *a, int n);
void cuda_logb_dd (double   *a, int n);

/* Trig / hyperbolic / angle / rounding wrappers — same shape and
   in-place contract as the transcendental block above. Integer
   inputs are promoted to fp32/fp64 upstream (rounding ops are no-ops
   on int, so they short-circuit before dispatch). */
#define DECLARE_UNOP_FP_WRAPPERS(OP)                                             \
    void cuda_##OP##_f16(uint16_t *a, int n);                                    \
    void cuda_##OP##_f32(float    *a, int n);                                    \
    void cuda_##OP##_f64(double   *a, int n);                                    \
    void cuda_##OP##_dd (double   *a, int n);

DECLARE_UNOP_FP_WRAPPERS(sin)
DECLARE_UNOP_FP_WRAPPERS(cos)
DECLARE_UNOP_FP_WRAPPERS(tan)
DECLARE_UNOP_FP_WRAPPERS(arcsin)
DECLARE_UNOP_FP_WRAPPERS(arccos)
DECLARE_UNOP_FP_WRAPPERS(arctan)
DECLARE_UNOP_FP_WRAPPERS(sinh)
DECLARE_UNOP_FP_WRAPPERS(cosh)
DECLARE_UNOP_FP_WRAPPERS(tanh)
DECLARE_UNOP_FP_WRAPPERS(arcsinh)
DECLARE_UNOP_FP_WRAPPERS(arccosh)
DECLARE_UNOP_FP_WRAPPERS(arctanh)
DECLARE_UNOP_FP_WRAPPERS(degrees)
DECLARE_UNOP_FP_WRAPPERS(radians)
DECLARE_UNOP_FP_WRAPPERS(rint)
DECLARE_UNOP_FP_WRAPPERS(trunc)
DECLARE_UNOP_FP_WRAPPERS(floor)
DECLARE_UNOP_FP_WRAPPERS(ceil)

#undef DECLARE_UNOP_FP_WRAPPERS

/* Precision-aware round (`round(x, decimals)`). Distinct signature from the
   wrappers above — it carries the `decimals` precision. The factor is
   computed on the host inside each wrapper; `decimals` may be negative. */
void cuda_round_f16(uint16_t *a, int decimals, int n);
void cuda_round_f32(float    *a, int decimals, int n);
void cuda_round_f64(double   *a, int decimals, int n);
void cuda_round_dd (double   *a, int decimals, int n);

void cuda_clip_i8 (int8_t   *a, int8_t   lo, int8_t   hi, int n);
void cuda_clip_u8 (uint8_t  *a, uint8_t  lo, uint8_t  hi, int n);
void cuda_clip_i16(int16_t  *a, int16_t  lo, int16_t  hi, int n);
void cuda_clip_u16(uint16_t *a, uint16_t lo, uint16_t hi, int n);
void cuda_clip_i32(int32_t  *a, int32_t  lo, int32_t  hi, int n);
void cuda_clip_u32(uint32_t *a, uint32_t lo, uint32_t hi, int n);
void cuda_clip_i64(int64_t  *a, int64_t  lo, int64_t  hi, int n);
void cuda_clip_u64(uint64_t *a, uint64_t lo, uint64_t hi, int n);
void cuda_clip_f16(uint16_t *a, float    lo, float    hi, int n);
void cuda_clip_f32(float    *a, float    lo, float    hi, int n);
void cuda_clip_f64(double   *a, double   lo, double   hi, int n);
/* DD clip: lo/hi each split into (hi, lo) doubles. */
void cuda_clip_dd (double   *a, double lo_hi, double lo_lo,
                                double hi_hi, double hi_lo, int n);

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