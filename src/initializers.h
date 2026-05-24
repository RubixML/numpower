#ifndef PHPSCI_NDARRAY_INITIALIZERS_H
#define PHPSCI_NDARRAY_INITIALIZERS_H

#include <Zend/zend_types.h>
#include "ndarray.h"

NDArray* NDArrayFactory_CreateFromDoubleScalar(double scalar);


NDArray* Create_NDArray(int* shape, int ndim, const char* type, int device);
NDArray* Create_NDArray_FromZval(zval* php_object);
NDArray* NDArray_FromNDArray(NDArray *target, int buffer_offset, int* shape, int* strides, const int* ndim);
NDArray* NDArray_Zeros(int *shape, int ndim, const char *type, int device);
NDArray* NDArray_Ones(int *shape, int ndim, const char *type, int device);
NDArray* NDArray_Identity(int size, const char *type, int device);

/**
 * @brief Arithmetic kind used to encode the (loc, scale) pair for the
 *        Gaussian sampler.
 *
 * Selected by the requested dtype: `float128` carries loc/scale in fp128
 * so wide-range parameters keep full 113-bit (or DD-equivalent)
 * precision; `uint64` uses native unsigned 64-bit arithmetic so means
 * past 2^53 stay exact; every other dtype uses double, which represents
 * each smaller dtype's range without loss.
 */
typedef enum {
    NDARRAY_NORMAL_KIND_DOUBLE = 0,
    NDARRAY_NORMAL_KIND_FP128  = 1,
    NDARRAY_NORMAL_KIND_UINT64 = 2
} NDArrayNormalKind;

/**
 * @brief Discriminated (loc, scale) parameter set for the Gaussian sampler.
 *
 * `kind` selects the active union arm; the others are not initialised.
 * Construction is handled by `PHP_METHOD(NumPower, normal)`.
 */
typedef struct {
    NDArrayNormalKind kind;
    union {
        struct { double          loc, scale; } d;
        struct { ndarray_fp128_t loc, scale; } f128;
        struct { uint64_t        loc, scale; } u64;
    } v;
} NDArrayNormalSpec;

/**
 * @brief Map a canonical dtype string to its normal-sampler arithmetic kind.
 *
 * @param[in] type Canonical dtype string.
 * @return one of NDARRAY_NORMAL_KIND_*.
 */
NDArrayNormalKind NDArray_NormalKindFor(const char *type);

NDArray* NDArray_Normal(const NDArrayNormalSpec *spec, int *shape, int ndim,
                        const char *type, int device);
NDArray* NDArray_TruncatedNormal(const NDArrayNormalSpec *spec, int *shape,
                                 int ndim, const char *type, int device);
NDArray* NDArray_StandardNormal(int* shape, int ndim);
NDArray* NDArray_Poisson(double lam, int* shape, int ndim);
NDArray* NDArray_Uniform(double low, double high, int* shape, int ndim);
NDArray* NDArray_Diag(NDArray *a, const char *type, int device);
NDArray* NDArray_FillFloat(NDArray *a, float fill_value);
NDArray* NDArray_Full(int *shape, int ndim, const char *type, int device,
                      const char *encoded);
NDArray* NDArray_CreateFromDoubleScalar(double scalar);
NDArray* NDArray_CreateFromLongScalar(long scalar);
int* Generate_Strides(const int* dimensions, int dimensions_size, int elsize);
NDArray* NDArray_CreateFromFloatScalar(float scalar);
NDArray* NDArray_Empty(int *shape, int ndim, const char *type, int device);
/**
 * @brief Arithmetic kind used to compute an arange sequence.
 *
 * Selected by the requested dtype: `float128` uses fp128 arithmetic so the
 * full 113-bit (or DD-equivalent) range is preserved; `int64` / `uint64`
 * use native 64-bit integer arithmetic so values past 2^53 stay exact;
 * everything else uses double, which represents every smaller dtype's
 * range exactly.
 */
typedef enum {
    NDARRAY_ARANGE_KIND_DOUBLE = 0,
    NDARRAY_ARANGE_KIND_FP128  = 1,
    NDARRAY_ARANGE_KIND_INT64  = 2,
    NDARRAY_ARANGE_KIND_UINT64 = 3
} NDArrayArangeKind;

/**
 * @brief Discriminated arange parameter set.
 *
 * Carries the (start, stop, step) triple in whichever native precision
 * the dtype requires. `kind` is the active union arm.
 */
typedef struct {
    NDArrayArangeKind kind;
    union {
        struct { double          start, stop, step; } d;
        struct { ndarray_fp128_t start, stop, step; } f128;
        struct { int64_t         start, stop, step; } i64;
        struct { uint64_t        start, stop, step; } u64;
    } v;
} NDArrayArangeSpec;

NDArray* NDArray_Arange(const NDArrayArangeSpec *spec,
                        const char *type, int device);

/**
 * @brief Map a canonical dtype string to its arange arithmetic kind.
 *
 * @param[in] type Canonical dtype string.
 * @return one of NDARRAY_ARANGE_KIND_*.
 */
NDArrayArangeKind NDArray_ArangeKindFor(const char *type);
NDArray* NDArray_Binomial(int *shape, int ndim, int n, float p);
NDArray* NDArray_EmptyLike(NDArray *a);
NDArray* NDArray_FromNDArrayBase(NDArray *target, char *data_ptr, int* shape, int* strides, const int ndim);

NDArray* NDArray_FillDouble(NDArray *a, double fill_value);
NDArray* NDArray_FillFloat128(NDArray *a, ndarray_fp128_t fill_value);

#ifdef __cplusplus
extern "C" {
#endif
NDArray *NDArray_Copy(NDArray *a, int device);
#ifdef __cplusplus
}
#endif

#endif //PHPSCI_NDARRAY_INITIALIZERS_H
