#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "linalg.h"
#ifndef _MSC_VER
#include "../../config.h"
#endif
#include "../initializers.h"
#include "../types.h"
#include "../manipulation.h"
#include "arithmetics.h"
#include "../iterators.h"
#include "../gpu_alloc.h"
#include "../indexing.h"

#ifdef HAVE_LAPACKE
/* OpenBLAS's lapack.h defaults lapack_complex_float / lapack_complex_double
   to "float _Complex" / "double _Complex" via <complex.h>. MSVC's <complex.h>
   only ships the _Fcomplex / _Dcomplex struct types — the C99 _Complex
   keyword isn't available — so those declarations cascade into syntax errors
   at the first complex-using LAPACK prototype. We never call any complex
   LAPACK routines, so opaque struct typedefs with the same layout suffice
   for the declarations to parse. The same struct layout is what
   MSVC's own _Fcomplex / _Dcomplex use, so even if a complex routine were
   added later the ABI would line up. */
#  ifdef _MSC_VER
#    define LAPACK_COMPLEX_CUSTOM
typedef struct { float  real, imag; } lapack_complex_float;
typedef struct { double real, imag; } lapack_complex_double;
#  endif
#include <lapacke.h>
#endif

#ifdef HAVE_LAPACKE_MKL
#include <mkl/mkl.h>
#endif

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda/cuda_math.h"
#include "../gpu_alloc.h"
#endif

#if HAVE_AVX2
#include <immintrin.h>
#endif

/**
 * Double type (float64) matmul
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_FMatmul(NDArray *a, NDArray *b) {
    int* output_shape = emalloc(sizeof(int) * 2);
    output_shape[0] = NDArray_SHAPE(a)[0];
    output_shape[1] = NDArray_SHAPE(b)[1];

    int m = NDArray_SHAPE(a)[0];
    int k = NDArray_SHAPE(a)[1];
    int n = NDArray_SHAPE(b)[1];
    char *type = NDArray_TYPE(a);

    if (is_type(type, NDARRAY_TYPE_FLOAT64)) {
        if (is_type(NDArray_TYPE(b), NDARRAY_TYPE_FLOAT64) == 0) {
            zend_throw_error(NULL, "matmul requires both operands to have the same float dtype");
            efree(output_shape);
            return NULL;
        }
        NDArray* result = NDArray_Zeros(output_shape, 2, NDARRAY_TYPE_FLOAT64, NDArray_DEVICE(a));
        if (result == NULL) {
            efree(output_shape);
            return NULL;
        }
        double alpha = 1.0;
        double beta = 0.0;
        if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            static cublasHandle_t handle = NULL;
            static bool handle_initialized = false;

            if (!handle_initialized) {
                cublasCreate(&handle);
                handle_initialized = true;
            }

            cublasDgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                NDArray_F64DATA(b), n,
                NDArray_F64DATA(a), k,
                &beta,
                (double*)NDArray_F64DATA(result), n
            );
#endif
        } else {
            cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    alpha,
                    NDArray_F64DATA(a), k,
                    NDArray_F64DATA(b), n,
                    beta,
                    NDArray_F64DATA(result), n
            );
        }
        return result;
    }

    NDArray* result = NDArray_Zeros(output_shape, 2, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));

    float alpha = 1.0f;
    float beta = 0.0f;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        static cublasHandle_t handle = NULL;
        static bool handle_initialized = false;

        if (!handle_initialized) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
            handle_initialized = true;
        }

        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            NDArray_F32DATA(b), n,
            NDArray_F32DATA(a), k,
            &beta,
            (float*)NDArray_F32DATA(result), n
        );

#endif
    } else {
        cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                NDArray_F32DATA(a), k,
                NDArray_F32DATA(b), n,
                beta,
                NDArray_F32DATA(result), n
        );
    }

    return result;
}

void
computeSVDFloat(float* A, int m, int n, float* U, float* S, float* V) {
    int lda = n;
    int ldu = m;
    int ldvt = n;

    int info;

    info = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, A, lda, S, U, ldu, V, ldvt);

    if (info > 0) {
        printf("SVD computation failed.\n");
        return;
    }

}

void
computeSVDDouble(double* A, int m, int n, double* U, double* S, double* V) {
    int lda = n;
    int ldu = m;
    int ldvt = n;
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', m, n, A, lda, S, U, ldu, V, ldvt);
    if (info > 0) {
        printf("SVD computation failed.\n");
    }
}

#ifdef HAVE_CUBLAS
void
computeSVDFloatGPU(float* A, int m, int n, float* U, float* S, float* V) {
    cuda_svd_float(A, U, V, S, m, n);
}
#endif

/**
 * @return
 */
NDArray**
NDArray_SVD(NDArray *target) {
    NDArray **rtns;
    NDArray *target_ptr = target;
    NDArray *rtn_s, *rtn_u, *rtn_v;
    double *Ud, *Sd, *Vd;
    float *output_data;
    float  *Uf, *Sf, *Vf;
    int *U_shape, *S_shape, *V_shape;
    int smallest_dim = -1;
    int is_double = is_type(NDArray_TYPE(target), NDARRAY_TYPE_FLOAT64);

    if (NDArray_NDIM(target) == 1) {
        zend_throw_error(NULL, "Array must be at least two-dimensional");
        return NULL;
    }

    rtns = emalloc(sizeof(NDArray*) * 3);

    for (int i = 0; i < NDArray_NDIM(target_ptr); i++) {
        if (smallest_dim == -1) {
            smallest_dim = NDArray_SHAPE(target_ptr)[i];
            continue;
        }
        if (smallest_dim > NDArray_SHAPE(target_ptr)[i]) {
            smallest_dim = NDArray_SHAPE(target_ptr)[i];
        }
    }

    // Allocate scratch buffers for the chosen dtype. In the GPU float32
    // path these live on-device (vmalloc); in the CPU path they are host.
    char  *U_buf, *S_buf, *V_buf;
    size_t U_elems = (size_t)NDArray_SHAPE(target)[0] * NDArray_SHAPE(target)[0];
    size_t V_elems = (size_t)NDArray_SHAPE(target)[1] * NDArray_SHAPE(target)[1];
    size_t S_elems = (size_t)smallest_dim;
    size_t A_elems = NDArray_NUMELEMENTS(target);
    if (is_double) {
        size_t e = sizeof(double);
        if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
            vmalloc((void**)&Ud, e * U_elems);
            vmalloc((void**)&Sd, e * S_elems);
            vmalloc((void**)&Vd, e * V_elems);
        } else {
            Ud = emalloc(e * U_elems);
            Sd = emalloc(e * S_elems);
            Vd = emalloc(e * V_elems);
        }
        U_buf = (char*)Ud; S_buf = (char*)Sd; V_buf = (char*)Vd;
    } else {
        if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
            vmalloc((void**)&Uf, sizeof(float) * U_elems);
            vmalloc((void**)&Sf, sizeof(float) * S_elems);
            vmalloc((void**)&Vf, sizeof(float) * V_elems);
            vmalloc((void**)&output_data, sizeof(float) * A_elems);
        } else {
            Uf = emalloc(sizeof(float) * U_elems);
            Sf = emalloc(sizeof(float) * S_elems);
            Vf = emalloc(sizeof(float) * V_elems);
            output_data = emalloc(sizeof(float) * A_elems);
        }
        U_buf = (char*)Uf; S_buf = (char*)Sf; V_buf = (char*)Vf;
    }

    if (is_double && NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        double *a_data = emalloc(sizeof(double) * A_elems);
        memcpy(a_data, NDArray_F64DATA(target_ptr), sizeof(double) * A_elems);
        computeSVDDouble(a_data, NDArray_SHAPE(target_ptr)[0], NDArray_SHAPE(target_ptr)[1], Ud, Sd, Vd);
        efree(a_data);
    } else if (NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        target_ptr = NDArray_Transpose(target, NULL);
        cudaMemcpy(output_data, NDArray_F32DATA(target_ptr), sizeof(float) * A_elems, cudaMemcpyDeviceToDevice);
        computeSVDFloatGPU(output_data, NDArray_SHAPE(target)[0], NDArray_SHAPE(target)[1], Uf, Sf, Vf);
#else
        return NULL;
#endif
    } else {
        memcpy(output_data, NDArray_F32DATA(target_ptr), sizeof(float) * A_elems);
        computeSVDFloat((float *) output_data, NDArray_SHAPE(target_ptr)[0], NDArray_SHAPE(target_ptr)[1], Uf, Sf, Vf);
    }

    if (NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_CPU && !is_double) {
        efree(output_data);
    }
    U_shape = emalloc(sizeof(int) * NDArray_NDIM(target_ptr));
    V_shape = emalloc(sizeof(int) * NDArray_NDIM(target_ptr));
    S_shape = emalloc(sizeof(int));
    S_shape[0] = smallest_dim;

    memcpy(U_shape, NDArray_SHAPE(target_ptr), sizeof(int) * NDArray_NDIM(target_ptr));
    U_shape[1] = NDArray_SHAPE(target_ptr)[0];

    memcpy(V_shape, NDArray_SHAPE(target_ptr), sizeof(int) * NDArray_NDIM(target_ptr));
    V_shape[0] = NDArray_SHAPE(target_ptr)[1];

    rtn_u = Create_NDArray(U_shape, NDArray_NDIM(target_ptr), NDArray_TYPE(target_ptr), NDArray_DEVICE(target_ptr));
    rtn_s = Create_NDArray(S_shape, 1, NDArray_TYPE(target_ptr), NDArray_DEVICE(target_ptr));
    rtn_v = Create_NDArray(V_shape, NDArray_NDIM(target_ptr), NDArray_TYPE(target_ptr), NDArray_DEVICE(target_ptr));

    rtn_u->data = U_buf;
    rtn_s->data = S_buf;
    rtn_v->data = V_buf;

    rtns[0] = rtn_u;
    rtns[1] = rtn_s;
    rtns[2] = rtn_v;

#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_GPU) {
        rtn_u->device = NDARRAY_DEVICE_GPU;
        rtn_s->device = NDARRAY_DEVICE_GPU;
        rtn_v->device = NDARRAY_DEVICE_GPU;
        vfree(output_data);
        NDArray_FREE(target_ptr);
    }
#endif

    return rtns;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Matmul(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        zend_throw_error(NULL, "Arrays must have the same shape. Broadcasting not implemented.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        return NDArray_Multiply_Float(a, b);
    }
    if (NDArray_NDIM(a) == 1 && NDArray_NDIM(b) == 1) {
        return NDArray_Dot(a, b);
    }

    if (NDArray_SHAPE(a)[NDArray_NDIM(a) - 1] != NDArray_SHAPE(b)[NDArray_NDIM(b) - 2]) {
        zend_throw_error(NULL, "Shape mismatch for matmul. cols(a) != rows(b)");
        return NULL;
    }

    if (NDArray_NDIM(a) > 2 && NDArray_NDIM(b) > 2) {
        zend_throw_error(NULL, "Stack of matrices not allowed");
        return NULL;
    }

    return NDArray_FMatmul(a, b);
}

/**
 * NDArray determinant
 *
 * @param a
 * @return
 */
NDArray*
NDArray_Det(NDArray *a) {
    int *new_shape = emalloc(sizeof(int));
    char *type = NDArray_TYPE(a);
    int is_double = is_type(type, NDARRAY_TYPE_FLOAT64);
    /* The determinant of a float64 matrix is a float64; of a float32 matrix,
       a float32. Preserve the source dtype so downstream consumers (and the
       PHP scalar produced by __toString) see the right width. */
    const char *result_type = is_double ? NDARRAY_TYPE_FLOAT64 : NDARRAY_TYPE_FLOAT32;
    NDArray *rtn = Create_NDArray(new_shape, 0, result_type, NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        rtn->device = NDARRAY_DEVICE_GPU;
        if (is_double) {
            vmalloc((void **)&rtn->data, sizeof(double));
            cuda_det_double(NDArray_F64DATA(a), NDArray_F64DATA(rtn), NDArray_SHAPE(a)[0]);
        } else {
            vmalloc((void **)&rtn->data, sizeof(float));
            cuda_det_float(NDArray_F32DATA(a), NDArray_F32DATA(rtn), NDArray_SHAPE(a)[0]);
        }
#else
        return NULL;
#endif
    } else if (is_double) {
        int info;
        int N = NDArray_SHAPE(a)[0];
        int* ipiv = (int*) emalloc(N * sizeof(int));
        double *matrix = emalloc(sizeof(double) * NDArray_NUMELEMENTS(a));
        rtn->data = emalloc(sizeof(double));
        memcpy(matrix, NDArray_F64DATA(a), sizeof(double) * NDArray_NUMELEMENTS(a));
        // LU Decomposition using LAPACKE interface
        dgetrf_(&N, &N, matrix, &N, ipiv, &info);

        if (info != 0) {
            if (info > 0) {
                NDArray_F64DATA(rtn)[0] = 0.0;
                efree(ipiv);
                efree(matrix);
                return rtn;
            }
            printf("Error in LU decomposition. Code: %d\n", info);
            efree(ipiv);
            exit(1);
        }

        // Calculate determinant as product of diagonal elements
        double det = 1.0;
        for (int i = 0; i < N; i++) {
            det *= matrix[i* N + i];
        }

        // Account for the parity of the permutation
        int num_perm = 0;
        for (int i = 0; i < N; i++) {
            if (i + 1 != ipiv[i]) num_perm++;
        }
        if (num_perm % 2 != 0) det = -det;

        efree(ipiv);
        efree(matrix);
        NDArray_F64DATA(rtn)[0] = det;
    } else {
        int info;
        int N = NDArray_SHAPE(a)[0];
        int* ipiv = (int*) emalloc(N * sizeof(int));
        float *matrix = emalloc(sizeof(float) * NDArray_NUMELEMENTS(a));
        rtn->data = emalloc(sizeof(float));
        memcpy(matrix, NDArray_F32DATA(a), sizeof(float) * NDArray_NUMELEMENTS(a));
        // LU Decomposition using LAPACKE interface
        sgetrf_(&N, &N, matrix, &N, ipiv, &info);

        if (info != 0) {
            if (info > 0) {
                NDArray_F32DATA(rtn)[0] = 0.f;
                efree(ipiv);
                efree(matrix);
                return rtn;
            }
            printf("Error in LU decomposition. Code: %d\n", info);
            efree(ipiv);
            exit(1);
        }

        // Calculate determinant as product of diagonal elements
        float det = 1;
        for (int i = 0; i < N; i++) {
            det *= matrix[i* N + i];
        }

        // Account for the parity of the permutation
        int num_perm = 0;
        for (int i = 0; i < N; i++) {
            if (i + 1 != ipiv[i]) num_perm++;
        }
        if (num_perm % 2 != 0) det = -det;

        efree(ipiv);
        efree(matrix);
        NDArray_F32DATA(rtn)[0] = det;
    }
    return rtn;
}

/**
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_Inner(NDArray *nda, NDArray *ndb) {
    NDArray *rtn = NULL;

    if (NDArray_NDIM(nda) == 0 && NDArray_NDIM(ndb) == 0) {
        return NDArray_Multiply_Float(nda, ndb);
    }

    int i;
    int last_dim_a, last_dim_b;
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray must be in the same device.");
        return NULL;
    }

    last_dim_a = NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1];
    last_dim_b = NDArray_SHAPE(ndb)[NDArray_NDIM(ndb) - 1];
    if (last_dim_a != last_dim_b) {
        zend_throw_error(NULL, "Shape is not aligned to perform the inner product.");
        return NULL;
    }

    NDArray *mul = NDArray_Multiply_Float(nda, ndb);
    rtn = NDArray_CreateFromFloatScalar(NDArray_Sum_Float(mul));
    NDArray_FREE(mul);
    if (NDArray_NDIM(nda) > 1) {
        rtn->ndim = NDArray_NDIM(nda);
        rtn->dimensions = emalloc(sizeof(int) * NDArray_NDIM(nda));
        rtn->strides = emalloc(sizeof(int) * NDArray_NDIM(nda));
        for (i = 0; i < NDArray_NDIM(rtn); i++) {
            NDArray_SHAPE(rtn)[i] = 1;
            NDArray_STRIDES(rtn)[i] = NDArray_ELSIZE(rtn);
        }
    }
    return rtn;
}

/**
 * NDArray dot product
 *
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_Dot(NDArray *nda, NDArray *ndb) {
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(nda) == 1 && NDArray_NDIM(ndb) == 1) {
        return NDArray_Inner(nda, ndb);
    } else if (NDArray_NDIM(nda) == 2 && NDArray_NDIM(ndb) == 2) {
        return NDArray_Matmul(nda, ndb);
    } else if (NDArray_NDIM(nda) == 0 || NDArray_NDIM(ndb) == 0) {
        return NDArray_Multiply_Float(nda, ndb);
    } else if (NDArray_NDIM(nda) > 0 && NDArray_NDIM(ndb) == 1) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            int *rtn_shape = emalloc(sizeof(int) * (NDArray_NDIM(nda) - 1));
            copy(NDArray_SHAPE(nda), rtn_shape, NDArray_NDIM(nda) -1);
            NDArray *rtn = NDArray_Empty(rtn_shape, NDArray_NDIM(nda) - 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
            cuda_float_multiply_matrix_vector(NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1], NDArray_F32DATA(nda), NDArray_F32DATA(ndb),
                                              NDArray_F32DATA(rtn), NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 2], NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1]);
            return rtn;
#endif
        } else {
#ifdef HAVE_CBLAS
            int *rtn_shape = emalloc(sizeof(int) * (NDArray_NDIM(nda) - 1));
            copy(NDArray_SHAPE(nda), rtn_shape, NDArray_NDIM(nda) -1);
            NDArray *rtn = NDArray_Empty(rtn_shape, NDArray_NDIM(nda) - 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
            cblas_sgemv(CblasRowMajor, CblasNoTrans, NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 2], NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1], 1.0f, NDArray_F32DATA(nda), NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1],
                        NDArray_F32DATA(ndb), 1, 0.0f, NDArray_F32DATA(rtn), 1);
            return rtn;
#endif
        }
    } else if (NDArray_NDIM(nda) > 0 && NDArray_NDIM(ndb) >= 2) {
        // @todo Implement missing conditional
        zend_throw_error(NULL, "Not implemented");
        return NULL;
    }
    return NULL;
}

/**
 * L2 NORM
 *
 * @param target
 * @return
 */
NDArray*
NDArray_L2Norm(NDArray* target) {
    NDArray *rtn = NULL;
    NDArray **svd = NDArray_SVD(target);
    if (svd == NULL) {
        return NULL;
    }
    float max_svd = NDArray_Max(svd[1]);
    rtn = NDArray_CreateFromFloatScalar(max_svd);
    NDArray_FREE(svd[0]);
    NDArray_FREE(svd[1]);
    NDArray_FREE(svd[2]);
    efree(svd);
    return rtn;
}

/**
 * L1 NORM
 *
 * @param target
 * @return
 */
NDArray*
NDArray_L1Norm(NDArray* target) {
    NDArray *rtn = NULL;
    float max_value = FLT_MIN;
    float *results = emalloc(sizeof(float) * NDArray_SHAPE(target)[NDArray_NDIM(target) - 2]);
    NDArray *transposed = NDArray_Transpose(target, NULL);
    NDArray *ab = NDArray_Abs(transposed);
    NDArray_FREE(transposed);
    NDArray *slice;
    while(!NDArrayIterator_ISDONE(ab)) {
        slice = NDArrayIterator_GET(ab);
        results[ab->iterator->currentIndex] = NDArray_Sum_Float(slice);
        NDArray_FREE(slice);
        NDArrayIterator_NEXT(ab);
    }
    for (int i = 0; i < NDArray_SHAPE(target)[NDArray_NDIM(target) - 2]; i++) {
        if (max_value < results[i]) {
            max_value = results[i];
        }
    }
    efree(results);
    NDArray_FREE(ab);
    rtn = NDArray_CreateFromFloatScalar(max_value);
    return rtn;
}

/**
 * Matrix or vector norm
 *
 * Types
 *  INT_MAX - Frobenius norm
 *  0 - sum(x!=0) Only Vectors
 *  1 - max(sum(abs(x), axis=0))
 * -1 - min(sum(abs(x), axis=0))
 *  2 - 2-norm
 * -2 - smallest singular value
 * @param target
 * @return
 */
NDArray*
NDArray_Norm(NDArray* target, int type) {
    if (type == 1) {
        return NDArray_L1Norm(target);
    }
    if (type == 2) {
        return NDArray_L2Norm(target);
    }

    zend_throw_error(NULL, "NDArray_Norm: The provided norm `%d` is invalid", type);
    return NULL;
}

/**
 *
 * @param matrix
 * @param n
 * @return 1 if succeeded, 0 if failed
 */
int
matrixFloatInverse(float* matrix, int n) {
    int* ipiv = (int*)emalloc(n * sizeof(int)); // Pivot indices
    int info; // Status variable

    // Perform LU factorization
    sgetrf_(&n, &n, matrix, &n, ipiv, &info);
    if (info != 0) {
        zend_throw_error(NULL, "LU factorization failed. Unable to compute the matrix inverse.\n");
        efree(ipiv);
        return 0;
    }

    // Calculate the inverse. Workspace is heap-allocated: lwork = n*n can be
    // megabytes for moderate matrices, and MSVC doesn't accept C99 VLAs anyway.
    int lwork = n * n;
    float *work_query = (float *)emalloc((size_t)lwork * sizeof(float));
    sgetri_(&n, matrix, &n, ipiv, work_query, &lwork, &info);
    if (info != 0) {
        zend_throw_error(NULL, "Matrix inversion failed.\n");
        efree(work_query);
        efree(ipiv);
        return 0;
    }
    efree(work_query);
    efree(ipiv);
    return 1;
}

/**
 * Double-precision (float64) matrix inverse via LAPACK dgetrf/dgetri.
 *
 * Companion to matrixFloatInverse for float64 inputs. The float32 path
 * re-casts double buffers as float* and runs single-precision LAPACK,
 * which both misreads the 8-byte element layout and loses precision —
 * so float64 arrays must take this branch instead.
 *
 * @param matrix
 * @param n
 * @return 1 if succeeded, 0 if failed
 */
int
matrixDoubleInverse(double* matrix, int n) {
    int* ipiv = (int*)emalloc(n * sizeof(int)); // Pivot indices
    int info; // Status variable

    // Perform LU factorization
    dgetrf_(&n, &n, matrix, &n, ipiv, &info);
    if (info != 0) {
        zend_throw_error(NULL, "LU factorization failed. Unable to compute the matrix inverse.\n");
        efree(ipiv);
        return 0;
    }

    // Calculate the inverse. Workspace is heap-allocated: lwork = n*n can be
    // megabytes for moderate matrices, and MSVC doesn't accept C99 VLAs anyway.
    int lwork = n * n;
    double *work_query = (double *)emalloc((size_t)lwork * sizeof(double));
    dgetri_(&n, matrix, &n, ipiv, work_query, &lwork, &info);
    if (info != 0) {
        zend_throw_error(NULL, "Matrix inversion failed.\n");
        efree(work_query);
        efree(ipiv);
        return 0;
    }
    efree(work_query);
    efree(ipiv);
    return 1;
}

/**
 *
 * @param matrix
 * @param n
 * @return 1 if succeeded, 0 if failed
 */
int
matrixFloatLU(float* matrix, int n, float *p, float *l, float *u) {
    int i, j, k, maxIndex;
    float maxVal, tempVal;

    // Initialize L, U, and P matrices
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                l[i * n + j] = 1.0f;
                u[i * n + j] = matrix[i * n + j];
            } else {
                l[i * n + j] = 0.0f;
                u[i * n + j] = matrix[i * n + j];
            }
            p[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Perform LU decomposition with partial pivoting
    for (k = 0; k < n - 1; k++) {
        maxIndex = k;
        maxVal = u[k * n + k];

        // Find the row with the maximum value in the current column
        for (i = k + 1; i < n; i++) {
            if (u[i * n + k] > maxVal) {
                maxIndex = i;
                maxVal = u[i * n + k];
            }
        }

        // Swap rows in U matrix
        if (maxIndex != k) {
            for (j = 0; j < n; j++) {
                tempVal = u[k * n + j];
                u[k * n + j] = u[maxIndex * n + j];
                u[maxIndex * n + j] = tempVal;

                tempVal = p[k * n + j];
                p[k * n + j] = p[maxIndex * n + j];
                p[maxIndex * n + j] = tempVal;
            }
        }

        // Perform elimination in U matrix and store multipliers in L matrix
        for (i = k + 1; i < n; i++) {
            l[i * n + k] = u[i * n + k] / u[k * n + k];
            for (j = k; j < n; j++) {
                u[i * n + j] -= l[i * n + k] * u[k * n + j];
            }
        }
    }
}

/**
 * Double-precision (float64) LU with partial pivoting. Companion to
 * matrixFloatLU for float64 inputs.
 *
 * @param matrix   (in)  input matrix, row-major, n x n
 * @param n        size of the square matrix
 * @param p        (out) permutation matrix
 * @param l        (out) unit lower-triangular factor
 * @param u        (out) upper-triangular factor
 * @return 1
 */
int
matrixDoubleLU(double* matrix, int n, double *p, double *l, double *u) {
    int i, j, k, maxIndex;
    double maxVal, tempVal;

    // Initialize L, U, and P matrices
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                l[i * n + j] = 1.0;
                u[i * n + j] = matrix[i * n + j];
            } else {
                l[i * n + j] = 0.0;
                u[i * n + j] = matrix[i * n + j];
            }
            p[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform LU decomposition with partial pivoting
    for (k = 0; k < n - 1; k++) {
        maxIndex = k;
        maxVal = u[k * n + k];

        // Find the row with the maximum value in the current column
        for (i = k + 1; i < n; i++) {
            if (u[i * n + k] > maxVal) {
                maxIndex = i;
                maxVal = u[i * n + k];
            }
        }

        // Swap rows in U matrix
        if (maxIndex != k) {
            for (j = 0; j < n; j++) {
                tempVal = u[k * n + j];
                u[k * n + j] = u[maxIndex * n + j];
                u[maxIndex * n + j] = tempVal;

                tempVal = p[k * n + j];
                p[k * n + j] = p[maxIndex * n + j];
                p[maxIndex * n + j] = tempVal;
            }
        }

        // Perform elimination in U matrix and store multipliers in L matrix
        for (i = k + 1; i < n; i++) {
            l[i * n + k] = u[i * n + k] / u[k * n + k];
            for (j = k; j < n; j++) {
                u[i * n + j] -= l[i * n + k] * u[k * n + j];
            }
        }
    }
    return 1;
}

/**
 * Calculate the inverse of a square NDArray
 *
 * @param target
 * @return
 */
NDArray*
NDArray_Inverse(NDArray* target) {
    int info;
    NDArray *rtn = NDArray_Copy(target, NDArray_DEVICE(target));
    rtn->uuid = -1;
    if (NDArray_NDIM(target) != 2) {
        zend_throw_error(NULL, "Array must be at least two-dimensional");
        NDArray_FREE(rtn);
        return NULL;
    }

    if (NDArray_SHAPE(target)[0] != NDArray_SHAPE(target)[1]) {
        zend_throw_error(NULL, "Array must be square");
        NDArray_FREE(rtn);
        return NULL;
    }

    int n = NDArray_SHAPE(rtn)[0];
    int is_double = is_type(NDArray_TYPE(rtn), NDARRAY_TYPE_FLOAT64);
    int is_float64_compatible = is_double || is_type(NDArray_TYPE(rtn), NDARRAY_TYPE_FLOAT128);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        // CPU INVERSE CALL — dispatch on dtype so float64 (and float128, which
        // downcasts) runs in double precision. The float32 path re-casts the
        // 8-byte element buffers to float* and would silently misread the
        // data, so it must not run against float64 inputs.
        if (is_float64_compatible) {
            if (!is_double) {
                NDArray *f64 = NDArray_AsType(rtn, NDARRAY_TYPE_FLOAT64);
                NDArray_FREE(rtn);
                rtn = f64;
            }
            info = matrixDoubleInverse(NDArray_F64DATA(rtn), n);
            if (!info) {
                NDArray_FREE(rtn);
                return NULL;
            }
        } else {
            info = matrixFloatInverse(NDArray_F32DATA(rtn), n);
            if (!info) {
                NDArray_FREE(rtn);
                return NULL;
            }
        }
    } else {
        // GPU INVERSE CALL — dispatch on dtype so float64 uses the
        // double-precision cuSOLVER path. float128 on GPU falls back to the
        // float64 downcast path (the dd⇄fp128 conversion lives outside the
        // GPU, so we compute in fp64 and leave the dtype as the result).
#ifdef HAVE_CUBLAS
        if (is_float64_compatible) {
            NDArray *f64 = rtn;
            if (!is_double) {
                f64 = NDArray_AsType(rtn, NDARRAY_TYPE_FLOAT64);
                NDArray_FREE(rtn);
                rtn = f64;
            }
            cuda_matrix_double_inverse(NDArray_F64DATA(rtn), n);
        } else {
            cuda_matrix_float_inverse(NDArray_F32DATA(rtn), n);
        }
#else
        (void)is_float64_compatible;
#endif
    }

    return rtn;
}

/**
 * Calculate the inverse of a square NDArray
 *
 * @param target
 * @return
 */
NDArray**
NDArray_LU(NDArray* target) {
    if (NDArray_NDIM(target) != 2) {
        zend_throw_error(NULL, "Array must be at least two-dimensional");
        return NULL;
    }
    if (NDArray_SHAPE(target)[0] != NDArray_SHAPE(target)[1]) {
        zend_throw_error(NULL, "Array must be square");
        return NULL;
    }
    NDArray **rtns = emalloc(sizeof(NDArray*) * 3);
    int info;
    int is_double = is_type(NDArray_TYPE(target), NDARRAY_TYPE_FLOAT64);
    const char *factor_type = is_double ? NDARRAY_TYPE_FLOAT64 : NDARRAY_TYPE_FLOAT32;
    int *new_shape_p = emalloc(sizeof(int) * NDArray_NDIM(target));
    int *new_shape_l = emalloc(sizeof(int) * NDArray_NDIM(target));
    int *new_shape_u = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape_p, NDArray_SHAPE(target), sizeof(int) * (int)NDArray_NDIM(target));
    memcpy(new_shape_l, NDArray_SHAPE(target), sizeof(int) * (int)NDArray_NDIM(target));
    memcpy(new_shape_u, NDArray_SHAPE(target), sizeof(int) * (int)NDArray_NDIM(target));
    NDArray *copied = NDArray_Copy(target, NDArray_DEVICE(target));
    NDArray *p = NDArray_Empty(new_shape_p, NDArray_NDIM(target), factor_type, NDArray_DEVICE(target));
    NDArray *l = NDArray_Empty(new_shape_l, NDArray_NDIM(target), factor_type, NDArray_DEVICE(target));
    NDArray *u = NDArray_Empty(new_shape_u, NDArray_NDIM(target), factor_type, NDArray_DEVICE(target));

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        // CPU INVERSE CALL
        if (is_double) {
            info = matrixDoubleLU(NDArray_F64DATA(copied),
                                  NDArray_SHAPE(copied)[0],
                                  NDArray_F64DATA(p),
                                  NDArray_F64DATA(l),
                                  NDArray_F64DATA(u));
        } else {
            info = matrixFloatLU(NDArray_F32DATA(copied),
                                 NDArray_SHAPE(copied)[0],
                                 NDArray_F32DATA(p),
                                 NDArray_F32DATA(l),
                                 NDArray_F32DATA(u));
        }
        if (!info) {
            NDArray_FREE(copied);
            return NULL;
        }
    } else {
        // GPU INVERSE CALL
#ifdef HAVE_CUBLAS
        if (is_double) {
            cuda_double_lu(NDArray_F64DATA(copied), NDArray_F64DATA(l), NDArray_F64DATA(u), NDArray_F64DATA(p), NDArray_SHAPE(copied)[0]);
        } else {
            cuda_float_lu(NDArray_F32DATA(copied), NDArray_F32DATA(l), NDArray_F32DATA(u), NDArray_F32DATA(p), NDArray_SHAPE(copied)[0]);
        }
#endif
    }
    NDArray_FREE(copied);
    rtns[0] = p;
    rtns[1] = l;
    rtns[2] = u;
    return rtns;
}

/**
 * NDArray matrix rank
 *
 * @param target
 * @param tol
 * @return
 */
NDArray*
NDArray_MatrixRank(NDArray *target, float *tol) {
    float mtol;
    int rank = 0, i;
    NDArray *rtn;
    NDArray **svd = NDArray_SVD(target);
    float *singular_values;

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        singular_values = NDArray_F32DATA(svd[1]);
    } else {
#ifdef HAVE_CUBLAS
        singular_values = emalloc(sizeof(float) * NDArray_NUMELEMENTS(target));
        cudaMemcpy(singular_values, NDArray_F32DATA(svd[1]), sizeof(float) * NDArray_NUMELEMENTS(svd[1]), cudaMemcpyDeviceToHost);
#endif
    }
    int minMN = (NDArray_SHAPE(target)[NDArray_NDIM(target) - 2] < (NDArray_SHAPE(target)[NDArray_NDIM(target) - 1])) ? (NDArray_SHAPE(target)[NDArray_NDIM(target) - 2]) : (NDArray_SHAPE(target)[NDArray_NDIM(target) - 1]);

    // Set the tolerance if not provided
    if (tol == NULL) {
        float maxSingularValue = singular_values[0];
        for (i = 1; i < minMN; i++) {
            if (singular_values[i] > maxSingularValue) {
                maxSingularValue = singular_values[i];
            }
        }
        mtol = maxSingularValue * fmaxf((float)NDArray_SHAPE(target)[NDArray_NDIM(target) - 2], (float)NDArray_SHAPE(target)[NDArray_NDIM(target) - 1]) * FLT_EPSILON;
    } else {
        mtol = *tol;
    }

    for (i = 0; i < minMN; i++) {
        if (singular_values[i] > mtol) {
            rank++;
        }
    }

    NDArray_FREE(svd[0]);
    NDArray_FREE(svd[1]);
    NDArray_FREE(svd[2]);
    efree(svd);
    rtn = NDArray_CreateFromLongScalar((int)rank);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        efree(singular_values);
    }

    return rtn;
}

/**
 * NDArray::outer
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Outer(NDArray *a, NDArray *b) {
    if (NDArray_NDIM(a) != 1 || NDArray_NDIM(b) != 1) {
        zend_throw_error(NULL, "Invalid operation: NDArray::outer() requires both arrays to be 1-dimensional vectors.");
        return NULL;
    }

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "NDArray::outer() requires both arrays to be on the same device (CPU or GPU).");
        return NULL;
    }
    int *output_shape = emalloc(sizeof(int) * 2);
    output_shape[0] = NDArray_NUMELEMENTS(a);
    output_shape[1] = NDArray_NUMELEMENTS(b);
    NDArray *rtn = NDArray_Zeros(output_shape, 2, NDArray_TYPE(a), NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
#ifdef HAVE_CBLAS
        cblas_sger(CblasRowMajor, NDArray_NUMELEMENTS(a), NDArray_NUMELEMENTS(b), 1.0f, NDArray_F32DATA(a), 1, NDArray_F32DATA(b), 1,
                   NDArray_F32DATA(rtn), NDArray_NUMELEMENTS(b));
#endif
    } else {
#ifdef HAVE_CUBLAS
        cuda_calculate_outer_product(NDArray_NUMELEMENTS(a), NDArray_NUMELEMENTS(b), NDArray_F32DATA(a), NDArray_F32DATA(b),
                                     NDArray_F32DATA(rtn));
#endif
    }
    return rtn;
}

/**
 * NDArray::trace
 *
 * @return
 */
NDArray*
NDArray_Trace(NDArray *a) {
    NDArray* diagonal = NDArray_Diagonal(a, 0);
    if (diagonal == NULL) {
        return NULL;
    }
    float result = NDArray_Sum_Float(diagonal);
    NDArray_FREE(diagonal);
    return NDArray_CreateFromFloatScalar(result);
}

int
computeEigenvaluesAndEigenvectorsFloat(NDArray* array, NDArray* rightEigenvectors,
                                        NDArray* eigenvalues, NDArray *wivectors, NDArray *leftEigenvectors) {
    // Assuming 'array' contains the input square matrix
    int n = array->dimensions[0]; // Size of the square matrix

    // Compute eigenvalues and right eigenvectors using LAPACK function
    int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, NDArray_F32DATA(array), n,
                              NDArray_F32DATA(rightEigenvectors), NDArray_F32DATA(wivectors), NDArray_F32DATA(leftEigenvectors),
                              n, NDArray_F32DATA(eigenvalues), n);

    // Check if the computation was successful (info == 0)
    if (info != 0) {
        zend_throw_error(NULL, "Error computing eigenvalues and eigenvectors.\n");
        return 0;
    }
    return 1;
}

int
computeEigenvaluesAndEigenvectorsDouble(NDArray* array, NDArray* rightEigenvectors,
                                        NDArray* eigenvalues, NDArray *wivectors, NDArray *leftEigenvectors) {
    int n = array->dimensions[0];
    int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, NDArray_F64DATA(array), n,
                              NDArray_F64DATA(rightEigenvectors), NDArray_F64DATA(wivectors), NDArray_F64DATA(leftEigenvectors),
                              n, NDArray_F64DATA(eigenvalues), n);
    if (info != 0) {
        zend_throw_error(NULL, "Error computing eigenvalues and eigenvectors.\n");
        return 0;
    }
    return 1;
}

/**
 * NDArray::eig
 *
 * @param a
 * @return
 */
NDArray**
NDArray_Eig(NDArray *a) {
    if (NDArray_NDIM(a) != 2 || NDArray_SHAPE(a)[0] != NDArray_SHAPE(a)[1]) {
        zend_throw_error(NULL, "Error: Input matrix is not square.\n");
        return NULL;
    }
    NDArray **rtn = emalloc(sizeof(NDArray*) * 2);
    NDArray* eigenvalues, *rightEigenvectors, *wivectors, *leftEigenvectors;
    int *eigenvalues_shape, *rightEigenvectors_shape, *wivectors_shape, *leftEigenvectors_shape;

    rightEigenvectors_shape = emalloc(sizeof(int));
    rightEigenvectors_shape[0] = NDArray_SHAPE(a)[0];

    rightEigenvectors = NDArray_Zeros(rightEigenvectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU && is_type(NDArray_TYPE(a), NDARRAY_TYPE_FLOAT64)) {
        wivectors_shape = emalloc(sizeof(int));
        leftEigenvectors_shape = emalloc(sizeof(int));
        eigenvalues_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        wivectors_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues_shape[1] = NDArray_SHAPE(a)[0];
        leftEigenvectors_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues = NDArray_Zeros(eigenvalues_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        wivectors = NDArray_Zeros(wivectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
        leftEigenvectors = NDArray_Zeros(leftEigenvectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
        if (!computeEigenvaluesAndEigenvectorsDouble(a, rightEigenvectors, eigenvalues, wivectors, leftEigenvectors)) {
            efree(rtn);
            return NULL;
        }
        NDArray_FREE(leftEigenvectors);
        NDArray_FREE(wivectors);
    } else if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        wivectors_shape = emalloc(sizeof(int));
        leftEigenvectors_shape = emalloc(sizeof(int));
        eigenvalues_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        wivectors_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues_shape[1] = NDArray_SHAPE(a)[0];
        leftEigenvectors_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues = NDArray_Zeros(eigenvalues_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        wivectors = NDArray_Zeros(wivectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
        leftEigenvectors = NDArray_Zeros(leftEigenvectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
        if (!computeEigenvaluesAndEigenvectorsFloat(a, rightEigenvectors, eigenvalues, wivectors, leftEigenvectors)) {
            efree(rtn);
            return NULL;
        }
        NDArray_FREE(leftEigenvectors);
        NDArray_FREE(wivectors);
    } else {
#ifdef HAVE_CUBLAS
        efree(rtn);
        NDArray_FREE(rightEigenvectors);
        zend_throw_error(NULL, "GPU eig currently unavailable");
        return NULL;
        //eigenvalues = NDArray_Copy(a, NDArray_DEVICE(a));
        //cuda_matrix_eig_float(NDArray_F32DATA(eigenvalues), NDArray_SHAPE(a)[0], NDArray_F32DATA(rightEigenvectors));
#endif
    }
    rtn[0] = rightEigenvectors;
    rtn[1] = eigenvalues;
    return rtn;
}

/**
 * NDArray::lstsq
 * @todo Implement GPU
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Lstsq(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::lstsq not implemented for GPU");
        return NULL;
    }

    // Check if input matrices have compatible dimensions
    if (NDArray_NDIM(a) != 2 || NDArray_NDIM(b) != 2 || NDArray_SHAPE(a)[0] != NDArray_SHAPE(b)[0]) {
        zend_throw_error(NULL, "Invalid dimensions to calculate lstsq, both arrays must have 2 dimensions and $b must contain the same amount of rows as $a");
        return NULL;
    }

    int is_double = is_type(NDArray_TYPE(a), NDARRAY_TYPE_FLOAT64);
    if (is_type(NDArray_TYPE(b), NDARRAY_TYPE_FLOAT64) != is_double) {
        zend_throw_error(NULL, "ndarray::lstsq requires both arrays to have matching float dtypes (both float32 or both float64)");
        return NULL;
    }

    int m = a->dimensions[0]; // Number of rows of the coefficient matrix A
    int n = a->dimensions[1]; // Number of columns of the coefficient matrix A
    int nrhs = b->dimensions[1]; // Number of right-hand sides (columns of B)

    int *out_shape = (int*)emalloc(2 * sizeof(int));
    out_shape[0] = n;
    out_shape[1] = nrhs;
    NDArray *x = NDArray_Zeros(out_shape, 2, is_double ? NDARRAY_TYPE_FLOAT64 : NDArray_TYPE(a), NDArray_DEVICE(a));
    if (is_double) {
        double *a_data = (double *) emalloc((size_t)m * n * sizeof(double));
        memcpy(a_data, a->data, (size_t)m * n * sizeof(double));
        double *b_data = (double *) emalloc((size_t)m * nrhs * sizeof(double));
        memcpy(b_data, b->data, (size_t)m * nrhs * sizeof(double));

        int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1], NDArray_SHAPE(b)[1],
                                 a_data,
                                 NDArray_SHAPE(a)[1], b_data, NDArray_SHAPE(b)[1]);

        if (info > 0) {
            zend_throw_error(NULL,
                             "The diagonal element %i of the triangular factor of $a is zero, so that $a does not have full rank.",
                             info);
            efree(a_data);
            efree(b_data);
            return NULL;
        }
        memcpy(NDArray_F64DATA(x), b_data, (size_t)n * nrhs * sizeof(double));
        efree(a_data);
        efree(b_data);
    } else if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        // Allocate memory and copy data for the coefficient matrix A
        float *a_data = (float *) emalloc(m * n * sizeof(float));
        memcpy(a_data, a->data, m * n * sizeof(float));

        // Allocate memory and copy data for the right-hand side matrix B
        float *b_data = (float *) emalloc(m * nrhs * sizeof(float));
        memcpy(b_data, b->data, m * nrhs * sizeof(float));

        int info = LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1], NDArray_SHAPE(b)[1],
                                 a_data,
                                 NDArray_SHAPE(a)[1], b_data, NDArray_SHAPE(b)[1]);

        if (info > 0) {
            zend_throw_error(NULL,
                             "The diagonal element %i of the triangular factor of $a is zero, so that $a does not have full rank.",
                             info);
            efree(a_data);
            efree(b_data);
            return NULL;
        }
        // Copy the result data to the output NDArray
        memcpy(NDArray_F32DATA(x), b_data, n * nrhs * sizeof(float));
        efree(a_data);
        efree(b_data);
    } else {
        /* cuda_lstsq_float was declared in cuda_math.h but never implemented
           in cuda_math.cu. On Linux it lurked as an undefined symbol resolved
           lazily by dlopen — calling NDArray_Lstsq on GPU data would have
           crashed at runtime. On Windows the strict linker refuses to build.
           Surface the missing GPU path as a clear error, mirroring how
           NDArray_Qr handles the same situation a few lines below. */
        zend_throw_error(NULL, "ndarray::lstsq not implemented for GPU");
        return NULL;
    }
    return x;
}

/**
 * NDArray::qr
 * @todo Implement GPU
 *
 * @param a
 * @return
 */
NDArray**
NDArray_Qr(NDArray *a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::qr not implemented for GPU");
        return NULL;
    }

    // Check if the input matrix is 2D
    if (a->ndim != 2) {
        return NULL;
    }

    int m = a->dimensions[0]; // Number of rows of the matrix A
    int n = a->dimensions[1]; // Number of columns of the matrix A

    // Ensure that m >= n for the QR factorization
    if (m < n) {
        return NULL;
    }

    // Allocate memory for the result matrices Q and R
    int *q_dimensions = (int*)emalloc(2 * sizeof(int));
    q_dimensions[0] = m;
    q_dimensions[1] = n;
    NDArray* q = NDArray_Zeros(q_dimensions, 2, NDArray_TYPE(a), NDArray_DEVICE(a));

    int *r_dimensions = (int*)emalloc(2 * sizeof(int));
    r_dimensions[0] = n;
    r_dimensions[1] = n;
    NDArray* r = NDArray_Zeros(r_dimensions, 2, NDArray_TYPE(a), NDArray_DEVICE(a));

    // Allocate memory and copy data for the matrix A
    float* a_data = (float*)emalloc(m * n * n * sizeof(float));
    memcpy(a_data, NDArray_F32DATA(a), m * n * sizeof(float));

    // Allocate memory for the workspace
    float* tau = (float*)emalloc(n * sizeof(float));
    int info;

    // Query the optimal workspace size
    info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a_data, n, tau);

    // Extract the upper triangular part of the matrix A to R
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            ((float*)r->data)[i * (r->strides[0]/NDArray_ELSIZE(r)) + j * (r->strides[1]/NDArray_ELSIZE(r))] = ((float*)a_data)[i * m + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < m; j++) {
            ((float*)q->data)[i * (q->strides[0]/NDArray_ELSIZE(q)) + j * (q->strides[1]/NDArray_ELSIZE(q))] = ((float*)a_data)[i * m + j];
        }
    }

    memcpy(NDArray_F32DATA(q), a_data, NDArray_NUMELEMENTS(a) * NDArray_ELSIZE(a));
    efree(tau);
    efree(a_data);
    NDArray **rtn = emalloc(sizeof(NDArray*) * 2);
    rtn[0] = q;
    rtn[1] = r;
    return rtn;
}

/**
 * NDArray::solve
 * @todo Implement GPU
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Solve(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Both NDArray must be in the same device.");
        return NULL;
    }
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::solve not implemented for GPU");
        return NULL;
    }
    // Check if input matrices are valid
    if (a == NULL || b == NULL) {
        return NULL;
    }

    // Check if input matrices are 2D and have compatible dimensions
    if (a->ndim != 2 || b->ndim != 2 || a->dimensions[0] != a->dimensions[1] || a->dimensions[0] != b->dimensions[0]) {
        zend_throw_error(NULL, "Incompatible shapes");
        return NULL;
    }

    int n = a->dimensions[0]; // Number of rows/columns of the square matrix A

    int is_double = is_type(NDArray_TYPE(a), NDARRAY_TYPE_FLOAT64);

    int *x_dimensions = (int*)emalloc(2 * sizeof(int));
    x_dimensions[0] = n;
    x_dimensions[1] = b->dimensions[1];
    NDArray *x = NDArray_Zeros(x_dimensions, 2, is_double ? NDARRAY_TYPE_FLOAT64 : NDArray_TYPE(a), NDArray_DEVICE(a));

    if (is_double) {
        double* a_data = (double*)emalloc((size_t)n * n * sizeof(double));
        memcpy(a_data, a->data, (size_t)n * n * sizeof(double));

        double* b_data = (double*)emalloc((size_t)n * x->dimensions[1] * sizeof(double));
        memcpy(b_data, b->data, (size_t)n * x->dimensions[1] * sizeof(double));

        int* ipiv = (int*)emalloc(n * sizeof(int));

        /* Row-major convention: lda (leading dim of row-major A) = number of
           columns of row-major A = n (A is square). ldb (leading dim of
           row-major B) = number of columns of row-major B = nrhs. Passing
           ldb=n (the number of rows) is the classic bug that produces a
           silently-wrong first element — see regression tests. */
        int nrhs = x->dimensions[1];
        int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a_data, n, ipiv, b_data, nrhs);
        if (info != 0) {
            zend_throw_error(NULL, "Solving linear system failed (LAPACKE_dgesv info=%d); is $a singular?", info);
            efree(a_data);
            efree(b_data);
            efree(ipiv);
            return NULL;
        }

        memcpy(x->data, b_data, (size_t)NDArray_NUMELEMENTS(b) * sizeof(double));

        efree(a_data);
        efree(b_data);
        efree(ipiv);
        return x;
    }

    // Allocate memory and copy data for the square matrix A
    float* a_data = (float*)emalloc(n * n * sizeof(float));
    memcpy(a_data, a->data, n * n * sizeof(float));

    // Allocate memory and copy data for the matrix B
    float* b_data = (float*)emalloc(n * x->dimensions[1] * sizeof(float));
    memcpy(b_data, b->data, n * x->dimensions[1] * sizeof(float));

    // Allocate memory for the pivot indices
    int* ipiv = (int*)emalloc(n * sizeof(int));

    int nrhs = x->dimensions[1];
    int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, a_data, n, ipiv, b_data, nrhs);
    if (info != 0) {
        zend_throw_error(NULL, "Solving linear system failed (LAPACKE_sgesv info=%d); is $a singular?", info);
        efree(a_data);
        efree(b_data);
        efree(ipiv);
        return NULL;
    }

    // Copy the result data to the output NDArray
    memcpy(x->data, b_data, NDArray_NUMELEMENTS(b) * sizeof(float));

    efree(a_data);
    efree(b_data);
    efree(ipiv);
    return x;
}

/**
 * NDArray::cond
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Cond(NDArray *a) {
    NDArray *a_norm = NDArray_L2Norm(a);
    NDArray *a_inv = NDArray_Inverse(a);
    NDArray *a_inv_norm = NDArray_L2Norm(a_inv);
    NDArray_FREE(a_inv);
    NDArray *rtn = NDArray_Multiply_Float(a_norm, a_inv_norm);
    NDArray_FREE(a_norm);
    NDArray_FREE(a_inv_norm);
    return rtn;
}

/**
 * NDArray::cholesky
 *
 * @todo Implement GPU
 * @param a
 * @return
 */
NDArray*
NDArray_Cholesky(NDArray *a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::cholesky not implemented for GPU");
        return NULL;
    }
    if (NDArray_NDIM(a) != 2 || NDArray_SHAPE(a)[0] != NDArray_SHAPE(a)[1]) {
        zend_throw_error(NULL, "NDArray_Cholesky: $a must be a square matrix.");
        return NULL;
    }

    NDArray *rtn = NDArray_Copy(a, NDArray_DEVICE(a));
    int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', NDArray_SHAPE(a)[0], NDArray_F32DATA(rtn), NDArray_SHAPE(a)[0]);

    if (info > 0) {
        NDArray_FREE(rtn);
        zend_throw_error(NULL, "Error calculating the cholesky decomposition. (Is $a not positive definite?)");
        return NULL;
    }
#if HAVE_AVX2
    int blockSize = 8; // AVX2 can process 8 single-precision floats at a time
    for (int i = 0; i < NDArray_SHAPE(a)[0]; i++) {
        // Perform AVX2 loop for blocks of 8 elements
        int j = i + 1;
        for (; j < NDArray_SHAPE(a)[0] - blockSize + 1; j += blockSize) {
            // Load 8 elements of the row (upper triangular elements) into an AVX register
            __m256 row_avx = _mm256_loadu_ps(&NDArray_F32DATA(rtn)[i * NDArray_SHAPE(a)[0] + j]);
            // Set all elements of the AVX register to 0
            __m256 zero_avx = _mm256_setzero_ps();
            // Store the 0s back into the upper triangular elements of the row
            _mm256_storeu_ps(&NDArray_F32DATA(rtn)[i * NDArray_SHAPE(a)[0] + j], zero_avx);
        }
        // Handle the remaining elements
        for (; j < NDArray_SHAPE(a)[0]; j++) {
            NDArray_F32DATA(rtn)[i * NDArray_SHAPE(a)[0] + j] = 0.0f;
        }
    }
#else
    for (int i = 0; i < NDArray_SHAPE(a)[0]; i++) {
        for (int j = i + 1; j < NDArray_SHAPE(a)[1]; j++) {
            NDArray_F32DATA(rtn)[i * NDArray_SHAPE(a)[0] + j] = 0.0f;
        }
    }
#endif

    return rtn;
}