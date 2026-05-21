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
void cuda_lstsq_float(float* A, int m, int n, float* B, int k, float* X);
NDArray* NDArrayMathGPU_ElementWise2F(NDArray* ndarray, ElementWiseFloatGPUOperation2F op, float val1, float val2);
NDArray* NDArrayMathGPU_ElementWise1F(NDArray* ndarray, ElementWiseFloatGPUOperation1F op, float val1);
void cuda_float_transpose(int tiledim, int blockrows, const float *d_in, float *d_out, int width, int height);
void cuda_float_positive(int nblocks, float *d_array);
void cuda_float_reciprocal(int nblocks, float *d_array);
void cuda_truncated_normal(float* d_data, int size, double loc, double scale);

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

#ifdef __cplusplus
}
#endif
#endif //PHPSCI_NDARRAY_CUDAMATH_H