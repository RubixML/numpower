#include "cuda_math.h"
#include <cuda_runtime.h>
#include "../../ndarray.h"
#include "../../gpu_alloc.h"
#include "../../initializers.h"
#include "../../debug.h"
#include <float.h>
#include <cusolverDn.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <time.h>
#include <type_traits>

#define CHECK_CUDA(func) do { \
  cudaError_t status = (func); \
  if (status != cudaSuccess) { \
    printf("CUDA API failed at line %d with error: %s\n", \
           __LINE__, cudaGetErrorString(status)); \
    return EXIT_FAILURE; \
  } \
} while (0)

#define CHECK_CUSOLVER(func) do { \
  cusolverStatus_t status = (func); \
  if (status != CUSOLVER_STATUS_SUCCESS) { \
    printf("cuSOLVER API failed at line %d with error: %d\n", \
           __LINE__, status); \
    return EXIT_FAILURE; \
  } \
} while (0)

/**
 * @brief Per-thread rejection-sample kernel for truncated Gaussian (float).
 *
 * Each thread initialises its own cuRAND state from `(seed, idx)` and
 * draws standard-normal samples until one lands in [-2, 2]; the accepted
 * sample is then scaled by `scale` and shifted by `loc` so the stored
 * value lies in `[loc - 2σ, loc + 2σ]`. The mean rejection rate at the
 * ±2σ window is ~4.55%, so each thread runs ~1.05 iterations on average
 * and is bounded by a hard cap (the implicit infinite loop is acceptable
 * for any scale > 0 — at most one in 10⁹ samples needs more than 50
 * iterations).
 *
 * @param[out] d_data Destination GPU float buffer.
 * @param[in]  size   Element count.
 * @param[in]  loc    Distribution mean (µ).
 * @param[in]  scale  Distribution stddev (σ); must be > 0.
 * @param[in]  seed   Per-call seed (see `cuda_normal_next_seed`).
 */
__global__ void truncatedNormalKernelF32(float* d_data, int size,
                                          float loc, float scale,
                                          unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState_t state;
    curand_init(seed, (unsigned long long)idx, 0, &state);
    float z;
    do {
        z = curand_normal(&state);
    } while (z < -2.0f || z > 2.0f);
    d_data[idx] = loc + scale * z;
}

/**
 * @brief Per-thread rejection-sample kernel for truncated Gaussian (double).
 *
 * Companion to `truncatedNormalKernelF32` for double-precision dtypes.
 * Uses `curand_normal_double` so the underlying samples carry 53-bit
 * precision before the affine transform.
 *
 * @param[out] d_data Destination GPU double buffer.
 * @param[in]  size   Element count.
 * @param[in]  loc    Distribution mean (µ).
 * @param[in]  scale  Distribution stddev (σ); must be > 0.
 * @param[in]  seed   Per-call seed.
 */
__global__ void truncatedNormalKernelF64(double* d_data, int size,
                                          double loc, double scale,
                                          unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState_t state;
    curand_init(seed, (unsigned long long)idx, 0, &state);
    double z;
    do {
        z = curand_normal_double(&state);
    } while (z < -2.0 || z > 2.0);
    d_data[idx] = loc + scale * z;
}

// CUDA kernel to calculate the median of a float* array
__global__ void findMedianKernelFloat(float* input, int size, float* median) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    if (globalIdx >= size)
        return;

    // Copy the data to shared memory
    sharedData[tid] = input[globalIdx];
    __syncthreads();

    // Perform parallel reduction to find the local median
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            float value1 = sharedData[index];
            float value2 = sharedData[index + stride];

            // Perform a simple swap to ensure value1 <= value2
            if (value1 > value2)
            {
                sharedData[index] = value2;
                sharedData[index + stride] = value1;
            }
        }
        __syncthreads();
    }

    // The median is the middle element of the sorted data
    if (tid == blockDim.x / 2)
        median[blockIdx.x] = sharedData[tid];
}

__global__ void calculateOuterProductFloat(float* a, float* b, int m, int n, float* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        result[row * n + col] = a[row] * b[col];
    }
}

__global__ void convolve2dSameFloatKernel(const float* a, const float* b,
                                      const int* shape_a, const int* shape_b,
                                      const int* strides_a, const int* strides_b,
                                      char boundary, float* output,
                                      float fill_value) {
    int a_height = shape_a[0];
    int a_width = shape_a[1];
    int b_height = shape_b[0];
    int b_width = shape_b[1];
    int stride_a_y = strides_a[0]/sizeof(float);
    int stride_a_x = strides_a[1]/sizeof(float);
    int stride_b_y = strides_b[0]/sizeof(float);
    int stride_b_x = strides_b[1]/sizeof(float);

    int output_height = a_height;
    int output_width = a_width;

    int padding_top = b_height / 2;
    int padding_left = b_width / 2;

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < output_height && x < output_width) {
        float sum = 0.0;

        for (int i = 0; i < b_height; i++) {
            for (int j = 0; j < b_width; j++) {
                int a_y = y + i - padding_top;
                int a_x = x + j - padding_left;

                if (boundary == 'f') {
                    if (a_y >= 0 && a_y < a_height && a_x >= 0 &&
                        a_x < a_width) {
                        sum += a[a_y * stride_a_y + a_x * stride_a_x] *
                               b[i * stride_b_y + j * stride_b_x];
                    } else {
                        sum += fill_value * b[i * stride_b_y + j * stride_b_x];
                    }
                } else if (boundary == 'w') {
                    int wrapped_y = (a_y + a_height) % a_height;
                    int wrapped_x = (a_x + a_width) % a_width;
                    sum += a[wrapped_y * stride_a_y + wrapped_x * stride_a_x] *
                           b[i * stride_b_y + j * stride_b_x];
                } else if (boundary == 's') {
                    int symm_y = (a_y < 0) ? -a_y - 1 : (a_y >= a_height) ? 2 * a_height - 1 - a_y : a_y;
                    int symm_x = (a_x < 0) ? -a_x - 1 : (a_x >= a_width) ? 2 * a_width - 1 - a_x : a_x;
                    sum += a[symm_y * stride_a_y + symm_x * stride_a_x] *
                           b[i * stride_b_y + j * stride_b_x];
                }
            }
        }

        output[y * output_width + x] = sum;
    }
}

__global__ void transposeCoalesced(const float* matIn, int height, int width, float* matTran)
{
    // Calculate the row and column index of the element
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within matrix bounds
    if (x < width && y < height) {
        int inputIdx = y * width + x;
        int outputIdx = x * height + y;
        matTran[outputIdx] = matIn[inputIdx];
    }
}


// CUDA kernel for LU decomposition
__global__ void luFloatDecompositionKernel(float *matrix, float *L, float *U, float *P, int size) {
    int i, k, maxIndex;
    float maxVal, tempVal;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        // Initialize L, U, and P matrices
        if (row == col) {
            L[row * size + col] = 1.0f;
            U[row * size + col] = matrix[row * size + col];
        } else {
            L[row * size + col] = 0.0f;
            U[row * size + col] = matrix[row * size + col];
        }
        P[row * size + col] = (row == col) ? 1.0f : 0.0f;

        // Perform LU decomposition with partial pivoting
        for (k = 0; k < size - 1; k++) {
            maxIndex = k;
            maxVal = U[k * size + k];

            // Find the row with the maximum value in the current column
            for (i = k + 1; i < size; i++) {
                if (U[i * size + k] > maxVal) {
                    maxIndex = i;
                    maxVal = U[i * size + k];
                }
            }

            // Swap rows in U matrix
            if (maxIndex != k) {
                tempVal = U[k * size + col];
                U[k * size + col] = U[maxIndex * size + col];
                U[maxIndex * size + col] = tempVal;

                tempVal = P[k * size + col];
                P[k * size + col] = P[maxIndex * size + col];
                P[maxIndex * size + col] = tempVal;
            }

            __syncthreads();

            // Perform elimination in U matrix and store multipliers in L matrix
            if (row > k && col >= k) {
                L[row * size + k] = U[row * size + k] / U[k * size + k];
                U[row * size + col] -= L[row * size + k] * U[k * size + col];
            }

            __syncthreads();
        }
    }
}

__global__ void roundToDecimalsFloatKernel(float* numbers, int decimals, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        float factor = powf(10, decimals);
        numbers[tid] = round(numbers[tid] * factor) / factor;
    }
}

__global__ void matrixL1NormFloatKernel(const float* matrix, float* result, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    while (idx < rows * cols) {
        sum += fabsf(matrix[idx]);
        idx += blockDim.x * gridDim.x;
    }

    atomicAdd(result, sum);
}

__global__ void matrixVectorMultiplyFloatKernel(float* a, float* b, float* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += a[row * cols + col] * b[col];
        }
        result[row] = sum;
    }
}

__global__ void compareArraysFloatKernel(const float* array1, const float* array2, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = (fabsf(array1[index] - array2[index]) <= 0.0000001f) ? 1.0f : 0.0f;
    }
}

__global__ void compareArraysNotEqualFloatKernel(const float* array1, const float* array2, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = (fabsf(array1[index] - array2[index]) <= 0.0000001f) ? 0.0f : 1.0f;
    }
}

__global__ void compareArraysGreaterFloatKernel(const float* array1, const float* array2, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = array1[index] > array2[index] ? 1.0f : 0.0f;
    }
}

__global__ void compareArraysGreaterEqualFloatKernel(const float* array1, const float* array2, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = array1[index] >= array2[index] ? 1.0f : 0.0f;
    }
}

__global__ void compareArraysLessFloatKernel(const float* array1, const float* array2, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = array1[index] < array2[index] ? 1.0f : 0.0f;
    }
}

__global__ void compareArraysLessEqualFloatKernel(const float* array1, const float* array2, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = array1[index] <= array2[index] ? 1.0f : 0.0f;
    }
}

__device__ float clipFloatValue(float value, float minVal, float maxVal) {
    return fminf(fmaxf(value, minVal), maxVal);
}

__global__ void clipFloatKernel(float* array, float minVal, float maxVal, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        array[index] = clipFloatValue(array[index], minVal, maxVal);
    }
}

__global__
void signFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        float value = d_array[index];
        d_array[index] = (value > 0) - (value < 0);
    }
}

__global__
void negateFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = -(d_array[index]);
    }
}

__global__
void positiveFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (d_array[index] < 0) {
            d_array[index] = -(d_array[index]);
        }
    }
}

__global__
void reciprocalFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = 1.0f / (d_array[index]);
    }
}


__device__
float sinc(float number) {
    if (number == 0.0) {
        return 1.0;
    } else {
        return sinf(M_PI * number) / (M_PI * number);
    }
}

__global__
void sincFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {

        d_array[index] = sinc(d_array[index]);
    }
}

__global__
void truncFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = truncf(d_array[index]);
    }
}

__device__
int roundFloatToNearestInt(float number) {
    float rounded = rintf(number);
    int floorInt = (int)floorf(number);

    // Check if the rounded value is halfway between two integers
    if (rounded - floorInt == 0.5 && ((int)rounded % 2 != 0)) {
        rounded -= 1.0;
    }

    return (int)rounded;
}

__global__
void rintFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = roundFloatToNearestInt(d_array[index]);
    }
}

__global__
void fixFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = truncf(d_array[index]);
    }
}

__global__
void ceilFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = ceilf(d_array[index]);
    }
}

__global__
void floorFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = floorf(d_array[index]);
    }
}

__global__
void arcsinhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = asinhf(d_array[index]);
    }
}

__global__
void arccoshFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = acoshf(d_array[index]);
    }
}

__global__
void arctanhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = atanhf(d_array[index]);
    }
}

__global__
void sinhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sinhf(d_array[index]);
    }
}

__global__
void coshFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = coshf(d_array[index]);
    }
}

__global__
void tanhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = tanhf(d_array[index]);
    }
}

__global__
void degreesFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = d_array[index] * (180.0 / 3.1415926535);
    }
}

__global__
void radiansFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = d_array[index] * (3.1415926535 / 180.0);
    }
}

__global__
void arcsinFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = asinf(d_array[index]);
    }
}

__global__
void arccosFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = acosf(d_array[index]);
    }
}

__global__
void arctanFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = atanf(d_array[index]);
    }
}

__global__
void absFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = fabsf(d_array[index]);
    }
}

__global__
void sinFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sinf(d_array[index]);
    }
}

__global__
void cosFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = cosf(d_array[index]);
    }
}

__global__
void tanFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = tanf(d_array[index]);
    }
}

__global__
void sqrtFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sqrtf(d_array[index]);
    }
}

__global__ void
add_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

__global__ void
subtract_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] - b[index];
    }
}

__global__ void
divide_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] / b[index];
    }
}

__global__ void
multiply_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] * b[index];
    }
}

__global__ void
fmodf_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = fmodf(a[index], b[index]);
    }
}

__global__ void
pow_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = powf(a[index], b[index]);
    }
}

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old_val_as_int = *address_as_int;
    int assumed;
    do {
        assumed = old_val_as_int;
        int max_val_as_int = __float_as_int(fmaxf(val, __int_as_float(old_val_as_int)));
        old_val_as_int = atomicCAS(address_as_int, assumed, max_val_as_int);
    } while (assumed != old_val_as_int);
    return __int_as_float(old_val_as_int);
}

/* Atomic float multiplication via CAS. CUDA has no built-in atomicMul for
   float, so we re-implement the standard pattern from the CUDA Programming
   Guide. Used by array_prod_float to merge per-block products into the
   global accumulator. */
__device__ float atomicMulFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old_val_as_int = *address_as_int;
    int assumed;
    do {
        assumed = old_val_as_int;
        int prod_val_as_int = __float_as_int(val * __int_as_float(old_val_as_int));
        old_val_as_int = atomicCAS(address_as_int, assumed, prod_val_as_int);
    } while (assumed != old_val_as_int);
    return __int_as_float(old_val_as_int);
}

/* Single-pass atomic-only max reduction.
 *
 * The previous `max_reduce_naive` did a two-stage shared-memory + warp
 * shuffle reduction. The final stage shuffled with mask 0xFFFFFFFF from
 * inside the divergent `if (tid < blockDim.x / warpSize)` block, which is
 * undefined on Volta+ — the inactive 24 lanes return garbage, and on
 * NVIDIA Ampere the garbage happened to be the seed slot's 0 bit pattern,
 * causing every positive-input max to come back as 0. The atomic-only
 * approach below is simpler and trades shared-memory bandwidth for a
 * tiny amount of atomic contention; for the typical reduction sizes
 * (n ≤ a few million) the difference is negligible. */
__global__ void
max_reduce_naive(float * result, float * data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicMaxFloat(result, data[i]);
    }
}

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old_val_as_int = *address_as_int;
    int assumed;
    do {
        assumed = old_val_as_int;
        int min_val_as_int = __float_as_int(fminf(val, __int_as_float(old_val_as_int)));
        old_val_as_int = atomicCAS(address_as_int, assumed, min_val_as_int);
    } while (assumed != old_val_as_int);
    return __int_as_float(old_val_as_int);
}

/* Single-pass atomic-only min reduction. See max_reduce_naive above for
 * the rationale — the original two-stage shuffle reduction was UB on
 * Volta+ and silently returned 0 for any positive-min input. */
__global__ void
min_reduce_naive(float * result, float * data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicMinFloat(result, data[i]);
    }
}

/* ── Double-precision atomics ───────────────────────────────────────────────
   CUDA's built-in atomicAdd(double*) requires CC 6.0+ (Pascal). For
   portability across older devices we implement all four ops via the
   same 64-bit CAS pattern. They are used by the per-dtype reduction
   kernels that always accumulate into a single `double` slot regardless
   of source dtype (so the C side can return a PHP `double` without
   dispatching on every reducer × dtype pair). */
__device__ double atomicAddDouble(double *address, double val) {
    unsigned long long *as_ull = (unsigned long long *) address;
    unsigned long long old = *as_ull, assumed;
    do {
        assumed = old;
        unsigned long long new_bits =
            (unsigned long long) __double_as_longlong(val + __longlong_as_double((long long) assumed));
        old = atomicCAS(as_ull, assumed, new_bits);
    } while (assumed != old);
    return __longlong_as_double((long long) old);
}

/* NaN-propagating min/max. Plain `fmin` / `fmax` follow IEEE 754-2008
   minNum/maxNum semantics — NaN is treated as "missing data" and the
   other operand wins. The CPU reducer in arithmetics.c propagates NaN
   (a single NaN element forces a NaN result), so the GPU path matches
   that convention via these helpers. */
__device__ inline double prop_fmin(double a, double b) {
    if (a != a) return a;   /* a is NaN */
    if (b != b) return b;   /* b is NaN */
    return fmin(a, b);
}
__device__ inline double prop_fmax(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return fmax(a, b);
}

__device__ double atomicMaxDouble(double *address, double val) {
    unsigned long long *as_ull = (unsigned long long *) address;
    unsigned long long old = *as_ull, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double((long long) assumed);
        unsigned long long new_bits =
            (unsigned long long) __double_as_longlong(prop_fmax(val, cur));
        old = atomicCAS(as_ull, assumed, new_bits);
    } while (assumed != old);
    return __longlong_as_double((long long) old);
}

__device__ double atomicMinDouble(double *address, double val) {
    unsigned long long *as_ull = (unsigned long long *) address;
    unsigned long long old = *as_ull, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double((long long) assumed);
        unsigned long long new_bits =
            (unsigned long long) __double_as_longlong(prop_fmin(val, cur));
        old = atomicCAS(as_ull, assumed, new_bits);
    } while (assumed != old);
    return __longlong_as_double((long long) old);
}

__device__ double atomicMulDouble(double *address, double val) {
    unsigned long long *as_ull = (unsigned long long *) address;
    unsigned long long old = *as_ull, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double((long long) assumed);
        unsigned long long new_bits =
            (unsigned long long) __double_as_longlong(val * cur);
        old = atomicCAS(as_ull, assumed, new_bits);
    } while (assumed != old);
    return __longlong_as_double((long long) old);
}

__global__
void array_equals_float(float *a, float *b, int *result, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        if (a[idx] != b[idx]) {
            atomicExch(result, 0); // If any element is not equal, set 'equal' to 0
        }
    }
}

__global__
void array_prod_float(float *a, float *result, int n) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float x = 1;
    if (i < n) x *= a[i];
    if (i + blockDim.x < n) x *= a[i + blockDim.x];
    sdata[tid] = x;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    /* Merge this block's product into the global accumulator. Must use
       atomicMulFloat (a CAS loop) — atomicAdd here would sum block
       products instead of multiplying them, and would also miss
       zero-containing inputs because the seed is 1 (1 + 0 = 1). */
    if (tid == 0) atomicMulFloat(result, sdata[0]);
}

__global__
void array_sum_float(float *a, float *result, int n) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float x = 0;
    if (i < n) x += a[i];
    if (i + blockDim.x < n) x += a[i + blockDim.x];
    sdata[tid] = x;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) atomicAdd(result, sdata[0]);
}

__global__ void array_sum_reduce_blocks(const double *a, double *block_results, int n) {
    extern __shared__ double sdataDouble[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = 0.0;

    if (i < n) sum += a[i];
    if (i + blockDim.x < n) sum += a[i + blockDim.x];

    sdataDouble[tid] = sum;
    __syncthreads();

    // Редукция в shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataDouble[tid] += sdataDouble[tid + s];
        }
        __syncthreads();
    }

    // Только первый поток сохраняет результат блока
    if (tid == 0) {
        block_results[blockIdx.x] = sdataDouble[0];
    }
}

__global__ void finalize_sum(const double *block_results, double *result, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += block_results[i];
    }
    *result = sum;
}

__global__
void fill_float_kernel(float* array, int n, float value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        array[idx] = value;
    }
}

__global__
void fill_float_kernel_double(double* array, int n, double value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        array[idx] = value;
    }
}

/* ──────────────────────────────────────────────────────────────────────────
   Typed CUDA kernels — keep GPU arrays on GPU for every supported dtype.

   For every binary op (add/sub/mul/div/mod/pow) we define a C++ template
   __global__ kernel and an extern "C" wrapper per dtype. The wrappers are
   the only API surface; the templates exist only inside the .cu file.

   Integer pow uses repeated-squaring fast exponentiation (exact); float pow
   uses pow() / powf() which already give native precision.

   Integer mod uses % (with C semantics: sign of result follows dividend).
   Float mod uses fmod() / fmodf().

   Division by zero on integer dtypes is undefined in C (just like CPU);
   on float dtypes it follows IEEE-754 (INF / NaN). We match this on GPU.
   ────────────────────────────────────────────────────────────────────────── */

template <typename T>
__device__ inline T tcuda_add(T a, T b) { return a + b; }
template <typename T>
__device__ inline T tcuda_sub(T a, T b) { return a - b; }
template <typename T>
__device__ inline T tcuda_mul(T a, T b) { return a * b; }
template <typename T>
__device__ inline T tcuda_div(T a, T b) { return a / b; }

/* Integer mod uses % with C semantics; protect against division by zero by
   returning 0 (matches PyTorch's defined behavior for integer x % 0). */
template <typename T>
__device__ inline T tcuda_mod_int(T a, T b) { return b == (T)0 ? (T)0 : a % b; }
__device__ inline double tcuda_mod_f(double a, double b) { return fmod(a, b); }
__device__ inline float  tcuda_mod_f(float a, float b)   { return fmodf(a, b); }

/* Integer pow: repeated squaring. Exact for all (a, b) where b >= 0; for
   b < 0 on signed types we return 0 (analogous to int division). */
template <typename T>
__device__ inline T tcuda_pow_int(T base, T exp) {
    if (exp < (T)0) return (T)0;
    T result = (T)1;
    T b = base;
    typename std::make_unsigned<T>::type e =
        (typename std::make_unsigned<T>::type)exp;
    while (e) {
        if (e & 1) result *= b;
        e >>= 1;
        if (e) b *= b;
    }
    return result;
}
template <typename T>
__device__ inline T tcuda_pow_uint(T base, T exp) {
    T result = (T)1;
    T b = base;
    while (exp) {
        if (exp & 1) result *= b;
        exp >>= 1;
        if (exp) b *= b;
    }
    return result;
}
__device__ inline double tcuda_pow_f(double a, double b) { return pow(a, b); }
__device__ inline float  tcuda_pow_f(float a, float b)   { return powf(a, b); }

/* Two-argument arctangent overloads — picks the dtype-correct libcudart
   intrinsic (`atan2` for double, `atan2f` for float) so a single templated
   kernel covers both float compute dtypes. */
__device__ inline double tcuda_atan2_f(double a, double b) { return atan2(a, b); }
__device__ inline float  tcuda_atan2_f(float a, float b)   { return atan2f(a, b); }

#define TYPED_BINOP_KERNEL(KERNEL_NAME, EXPR)                                       \
template <typename T>                                                                \
__global__ void KERNEL_NAME(const T *a, const T *b, T *out, int n) {                 \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                                   \
    if (i < n) { T x = a[i]; T y = b[i]; out[i] = (EXPR); }                          \
}

TYPED_BINOP_KERNEL(tcuda_add_kernel, x + y)
TYPED_BINOP_KERNEL(tcuda_sub_kernel, x - y)
TYPED_BINOP_KERNEL(tcuda_mul_kernel, x * y)
TYPED_BINOP_KERNEL(tcuda_div_kernel, x / y)
/* Element-wise atan2(x, y). Instantiated only for float / double (the
   float compute dtypes the host dispatcher routes here); fp128 has its
   own dd kernel below. */
TYPED_BINOP_KERNEL(tcuda_atan2_kernel, tcuda_atan2_f(x, y))

/* Specialised mod/pow kernels (need different impl for int vs float) */
template <typename T>
__global__ void tcuda_mod_int_kernel(const T *a, const T *b, T *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = tcuda_mod_int(a[i], b[i]);
}
__global__ void tcuda_mod_f64_kernel(const double *a, const double *b, double *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = fmod(a[i], b[i]);
}

template <typename T>
__global__ void tcuda_pow_signed_kernel(const T *a, const T *b, T *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = tcuda_pow_int(a[i], b[i]);
}
template <typename T>
__global__ void tcuda_pow_unsigned_kernel(const T *a, const T *b, T *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = tcuda_pow_uint(a[i], b[i]);
}
__global__ void tcuda_pow_f64_kernel(const double *a, const double *b, double *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = pow(a[i], b[i]);
}

/* __half (float16) kernels — promote to float for compute to retain accuracy
   matching CPU's float32 compute path. */
__global__ void tcuda_add_half_kernel(const __half *a, const __half *b, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}
__global__ void tcuda_sub_half_kernel(const __half *a, const __half *b, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) - __half2float(b[i]));
}
__global__ void tcuda_mul_half_kernel(const __half *a, const __half *b, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}
__global__ void tcuda_div_half_kernel(const __half *a, const __half *b, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) / __half2float(b[i]));
}
__global__ void tcuda_mod_half_kernel(const __half *a, const __half *b, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(fmodf(__half2float(a[i]), __half2float(b[i])));
}
__global__ void tcuda_pow_half_kernel(const __half *a, const __half *b, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(powf(__half2float(a[i]), __half2float(b[i])));
}

/* Generic typed fill. */
template <typename T>
__global__ void tcuda_fill_kernel(T *out, T value, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = value;
}
__global__ void tcuda_fill_half_kernel(__half *out, float value, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half(value);
}

/* GPU AsType cast kernels. Each pair (Src → Dst) is template-instantiated.
   Casts are value-preserving where the destination range fits the source;
   when not, C-style cast semantics apply (truncate / wrap), matching CPU. */
template <typename Src, typename Dst>
__global__ void tcuda_cast_kernel(const Src *in, Dst *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = (Dst)in[i];
}
template <typename Src>
__global__ void tcuda_cast_to_half_kernel(const Src *in, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __float2half((float)in[i]);
}
template <typename Dst>
__global__ void tcuda_cast_from_half_kernel(const __half *in, Dst *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = (Dst)__half2float(in[i]);
}
__global__ void tcuda_cast_half_to_half_kernel(const __half *in, __half *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = in[i];
}

/* ──────────────────────────────────────────────────────────────────────────
   GPU float128 emulation via double-double (TwoSum / Veltkamp split).

   Storage: each fp128 element on GPU occupies 16 bytes laid out as (hi, lo)
   where hi is the leading double and lo is the residual. Effective precision:
   ~106 bits (vs fp128's 113 bits). Conversion to/from native __float128
   happens host-side in NDArray_ToGPU / NDArray_ToCPU.

   References: Dekker (1971), Bailey & Hida (2001) "Algorithms for
   Quad-Double Precision Floating Point Arithmetic".
   ────────────────────────────────────────────────────────────────────────── */
struct dd_real { double hi; double lo; };

__device__ inline dd_real dd_make(double h, double l) { dd_real r; r.hi = h; r.lo = l; return r; }

/* TwoSum: a + b = s + e exactly, |e| << |s|. */
__device__ inline dd_real dd_two_sum(double a, double b) {
    double s  = a + b;
    double bb = s - a;
    double e  = (a - (s - bb)) + (b - bb);
    return dd_make(s, e);
}
/* TwoProd via FMA. CRITICAL: use `__dmul_rn` (round-to-nearest, NOT
   FMA-contracted) for the leading product so the residual `fma(a, b, -p)`
   captures the actual rounding error of `a * b`. NVCC's default `-fmad=true`
   silently contracts `a * b` into an FMA pattern when it sees a nearby
   FMA call, giving a different rounding than IEEE fp64 multiply and
   collapsing DD precision to fp64 (Hida/Li/Bailey two_prod's invariant
   `a*b == p + e` breaks if `p` isn't the correctly-rounded product). */
__device__ inline dd_real dd_two_prod(double a, double b) {
    double p = __dmul_rn(a, b);
    double e = fma(a, b, -p);
    return dd_make(p, e);
}

__device__ inline dd_real dd_add(dd_real a, dd_real b) {
    dd_real s = dd_two_sum(a.hi, b.hi);
    dd_real t = dd_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    dd_real s2 = dd_two_sum(s.hi, s.lo);
    s2.lo += t.lo;
    return dd_two_sum(s2.hi, s2.lo);
}
__device__ inline dd_real dd_neg(dd_real a) { return dd_make(-a.hi, -a.lo); }
__device__ inline dd_real dd_sub(dd_real a, dd_real b) { return dd_add(a, dd_neg(b)); }
__device__ inline dd_real dd_mul(dd_real a, dd_real b) {
    dd_real p = dd_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return dd_two_sum(p.hi, p.lo);
}
__device__ inline dd_real dd_div(dd_real a, dd_real b) {
    /* Long-form division: q1 = a.hi / b.hi, refine */
    double q1 = a.hi / b.hi;
    dd_real r1 = dd_mul(b, dd_make(q1, 0.0));
    dd_real diff = dd_sub(a, r1);
    double q2 = diff.hi / b.hi;
    dd_real r2 = dd_mul(b, dd_make(q2, 0.0));
    dd_real diff2 = dd_sub(diff, r2);
    double q3 = diff2.hi / b.hi;
    return dd_add(dd_add(dd_make(q1, 0.0), dd_make(q2, 0.0)), dd_make(q3, 0.0));
}

/* dd_pow uses x^y = exp(y * log(x)) via dd_exp and dd_log; for simplicity we
   use double-precision exp/log here and accept the precision degradation for
   pow specifically. Add/sub/mul/div retain full double-double precision. */
__device__ inline double dd_to_double(dd_real a) { return a.hi; }
__device__ inline dd_real dd_pow(dd_real a, dd_real b) {
    double r = pow(dd_to_double(a), dd_to_double(b));
    return dd_make(r, 0.0);
}
__device__ inline dd_real dd_mod(dd_real a, dd_real b) {
    /* fmod(a, b) = a - trunc(a / b) * b  (double-double precision) */
    dd_real q = dd_div(a, b);
    double q_trunc = trunc(q.hi);
    dd_real qt = dd_mul(b, dd_make(q_trunc, 0.0));
    return dd_sub(a, qt);
}
/* dd_atan2 degrades to fp64 — same contract as the dd unary transcendentals
   (sin/cos/atan, which all route dd → double → libm → dd). Full 113-bit
   atan2 needs the libquadmath CPU path; the GPU dd tier tops out at fp64. */
__device__ inline dd_real dd_atan2(dd_real a, dd_real b) {
    return dd_make(atan2(dd_to_double(a), dd_to_double(b)), 0.0);
}

#define DD_BINOP_KERNEL(NAME, OP)                                                   \
__global__ void NAME(const double *a, const double *b, double *out, int n) {        \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                                  \
    if (i < n) {                                                                    \
        dd_real x = dd_make(a[2*i], a[2*i+1]);                                      \
        dd_real y = dd_make(b[2*i], b[2*i+1]);                                      \
        dd_real r = OP(x, y);                                                       \
        out[2*i]   = r.hi;                                                          \
        out[2*i+1] = r.lo;                                                          \
    }                                                                               \
}

DD_BINOP_KERNEL(tcuda_add_dd_kernel, dd_add)
DD_BINOP_KERNEL(tcuda_sub_dd_kernel, dd_sub)
DD_BINOP_KERNEL(tcuda_mul_dd_kernel, dd_mul)
DD_BINOP_KERNEL(tcuda_div_dd_kernel, dd_div)
DD_BINOP_KERNEL(tcuda_pow_dd_kernel, dd_pow)
DD_BINOP_KERNEL(tcuda_mod_dd_kernel, dd_mod)
DD_BINOP_KERNEL(tcuda_atan2_dd_kernel, dd_atan2)

__global__ void tcuda_fill_dd_kernel(double *out, double hi, double lo, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[2*i]   = hi;
        out[2*i+1] = lo;
    }
}

/* ──────────────────────────────────────────────────────────────────────────
   Device-side DD unary helpers. Pair with the existing dd_add / dd_mul /
   dd_div / dd_neg primitives (above) to provide abs, sign, sqrt, rsqrt
   and a clip comparator used by the typed unary kernels.
   ────────────────────────────────────────────────────────────────────────── */

__device__ inline dd_real dd_abs(dd_real a) {
    return (a.hi < 0.0) ? dd_neg(a) : a;
}

__device__ inline int dd_cmp(dd_real a, dd_real b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return  1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return  1;
    return 0;
}

__device__ inline dd_real dd_sign(dd_real a) {
    /* PyTorch sign(NaN) = NaN. A DD-encoded NaN has `hi != hi`; the
       canonical IEEE-754 NaN test is the self-compare. Without this
       guard the fall-through would silently return 0 (every ordered
       compare against NaN is false). */
    if (a.hi != a.hi)                          return a;
    if (a.hi > 0.0)                            return dd_make( 1.0, 0.0);
    if (a.hi < 0.0)                            return dd_make(-1.0, 0.0);
    if (a.lo > 0.0)                            return dd_make( 1.0, 0.0);
    if (a.lo < 0.0)                            return dd_make(-1.0, 0.0);
    return dd_make(0.0, 0.0);
}

/* Single Newton refinement (one DD multiplication + one DD division) over a
   fp64 sqrt seed: matches the CPU `ndarray_dd_sqrt` bit-for-bit. */
__device__ inline dd_real dd_sqrt(dd_real a) {
    if (a.hi == 0.0 && a.lo == 0.0) return a;
    if (a.hi < 0.0) return dd_make(nan(""), 0.0);
    double y = sqrt(a.hi);
    dd_real y_dd  = dd_make(y, 0.0);
    dd_real y_sq  = dd_mul(y_dd, y_dd);
    dd_real diff  = dd_sub(a, y_sq);
    dd_real denom = dd_make(2.0 * y, 0.0);
    dd_real corr  = dd_div(diff, denom);
    return dd_add(y_dd, corr);
}

__device__ inline dd_real dd_recip(dd_real a) {
    return dd_div(dd_make(1.0, 0.0), a);
}

__device__ inline dd_real dd_rsqrt(dd_real a) {
    return dd_recip(dd_sqrt(a));
}

/* ── DD-precision transcendentals (device side) ─────────────────────────
   Identical algorithms to the CPU helpers in `src/dd_math.c`; the shared
   DD constants below are byte-equal to `DD_LN2` / `DD_LOG2_E` /
   `DD_LOG10_E` there so CPU↔GPU parity is preserved. Every step runs in
   DD arithmetic — no internal collapse to fp64 — so the result holds
   ~32 decimal digits of precision the same as the libquadmath path on
   the host. Each function mirrors the doxygen on its CPU twin in
   `src/dd_math.c`; see there for the per-function range-reduction and
   series rationale. */
__device__ inline dd_real dd_ln2(void)    { return dd_make(0.6931471805599453, 2.3190468138462996e-17); }
__device__ inline dd_real dd_log2e(void)  { return dd_make(1.4426950408889634, 2.0355273740931033e-17); }
__device__ inline dd_real dd_log10e(void) { return dd_make(0.4342944819032518, 1.0983196502167645e-17); }

/** @brief DD-precision exp(x); see ndarray_dd_exp in src/dd_math.c. */
__device__ inline dd_real dd_exp(dd_real a) {
    if (isnan(a.hi)) return a;
    if (isinf(a.hi)) return dd_make(a.hi > 0 ? a.hi : 0.0, 0.0);
    if (a.hi >  709.7827) return dd_make(INFINITY, 0.0);
    if (a.hi < -745.1332) return dd_make(0.0, 0.0);
    /* Range reduction: x = k·ln(2) + r, k = round(x/ln(2)). */
    double  k_d  = round(a.hi * 1.4426950408889634);
    int     k    = (int)k_d;
    dd_real k_dd = dd_make(k_d, 0.0);
    dd_real r = dd_sub(a, dd_mul(k_dd, dd_ln2()));

    /* Horner of 1 + r·(1 + r/2·(1 + r/3·(… + r/24))). 24 terms keep the
       worst-case remainder r²⁵/25! below DD epsilon (see CPU twin). */
    dd_real result = dd_make(1.0, 0.0);
    for (int i = 24; i >= 1; i--) {
        dd_real r_over_i = dd_div(r, dd_make((double)i, 0.0));
        result = dd_add(dd_make(1.0, 0.0), dd_mul(r_over_i, result));
    }

    /* Scale by 2^k: ldexp is exact (exponent-only op). */
    result.hi = ldexp(result.hi, k);
    result.lo = ldexp(result.lo, k);
    return result;
}

/** @brief DD-precision expm1(x); see ndarray_dd_expm1 in src/dd_math.c. */
__device__ inline dd_real dd_expm1(dd_real a) {
    if (isnan(a.hi)) return a;
    if (a.hi >= 0.5 || a.hi <= -0.5) {
        return dd_sub(dd_exp(a), dd_make(1.0, 0.0));
    }
    /* Horner of x·(1 + x/2·(1 + x/3·(…))) — 25 inner steps capture
       (0.5)^25/25! < DD ULP. */
    dd_real result = dd_make(1.0, 0.0);
    for (int i = 25; i >= 2; i--) {
        dd_real a_over_i = dd_div(a, dd_make((double)i, 0.0));
        result = dd_add(dd_make(1.0, 0.0), dd_mul(a_over_i, result));
    }
    return dd_mul(a, result);
}

/** @brief DD-precision natural log(x); see ndarray_dd_log in src/dd_math.c. */
__device__ inline dd_real dd_log(dd_real a) {
    if (isnan(a.hi)) return a;
    if (a.hi < 0.0)              return dd_make(nan(""), 0.0);
    if (a.hi == 0.0 && a.lo == 0.0) return dd_make(-INFINITY, 0.0);
    if (isinf(a.hi))             return dd_make(INFINITY, 0.0);

    int     e;
    double  m_hi = frexp(a.hi, &e);
    dd_real m    = dd_make(m_hi, ldexp(a.lo, -e));

    /* Bring m into [sqrt(0.5), sqrt(2)) so |u| ≤ ~0.172. */
    if (m.hi < 0.7071067811865476) {
        m  = dd_add(m, m);
        e -= 1;
    }

    const dd_real one = dd_make(1.0, 0.0);
    dd_real u   = dd_div(dd_sub(m, one), dd_add(m, one));
    dd_real u2  = dd_mul(u, u);

    /* 2·atanh(u) = 2·(u + u³/3 + u⁵/5 + … + u^(2N-1)/(2N-1)).
       For |u| ≤ ~0.172 (the post-shift range), the truncated-series
       error after the last term u^(2N-1)/(2N-1) is below DD epsilon
       (~10⁻³²) once 2N-1 ≥ ~41. Take 2N-1 = 51 for headroom and use
       full DD-precision reciprocal constants (`dd_div(one, k)`; an
       fp64 `1.0 / k` carries only ~16 digits and collapses the
       series back to fp64 precision). */
    dd_real sum = dd_div(one, dd_make(51.0, 0.0));
    for (int k = 49; k >= 1; k -= 2) {
        dd_real inv_k = dd_div(one, dd_make((double)k, 0.0));
        sum = dd_add(inv_k, dd_mul(u2, sum));
    }
    dd_real log_m = dd_mul(u, sum);
    log_m = dd_add(log_m, log_m);  /* · 2 */

    return dd_add(log_m, dd_mul(dd_make((double)e, 0.0), dd_ln2()));
}

/** @brief DD-precision log1p(x); see ndarray_dd_log1p in src/dd_math.c. */
__device__ inline dd_real dd_log1p(dd_real a) {
    if (isnan(a.hi)) return a;
    if (a.hi >= 0.5 || a.hi <= -0.5) {
        return dd_log(dd_add(dd_make(1.0, 0.0), a));
    }
    /* |a| ≤ 0.5: log1p(a) = 2·atanh(a/(2+a)). Forming a/(2+a) keeps a's
       lo limb (2+a never cancels), unlike forming 1+a which would round
       sub-fp64 information away. |u| ≤ 0.2 → 26-odd-term atanh ladder is
       below DD epsilon. Mirrors the CPU twin exactly for parity. */
    const dd_real one = dd_make(1.0, 0.0);
    dd_real u   = dd_div(a, dd_add(dd_make(2.0, 0.0), a));
    dd_real u2  = dd_mul(u, u);
    dd_real sum = dd_div(one, dd_make(51.0, 0.0));
    for (int k = 49; k >= 1; k -= 2) {
        dd_real inv_k = dd_div(one, dd_make((double)k, 0.0));
        sum = dd_add(inv_k, dd_mul(u2, sum));
    }
    dd_real r = dd_mul(u, sum);
    return dd_add(r, r);  /* · 2 */
}

/** @brief DD-precision exp2(x) = 2^x; see ndarray_dd_exp2 in src/dd_math.c. */
__device__ inline dd_real dd_exp2(dd_real a) {
    return dd_exp(dd_mul(a, dd_ln2()));
}

/** @brief DD-precision log2(x); see ndarray_dd_log2 in src/dd_math.c. */
__device__ inline dd_real dd_log2(dd_real a) {
    /* Power-of-2 exact short-circuit. */
    if (a.lo == 0.0 && isfinite(a.hi) && a.hi > 0.0) {
        int e;
        double m = frexp(a.hi, &e);
        if (m == 0.5) return dd_make((double)(e - 1), 0.0);
    }
    return dd_mul(dd_log(a), dd_log2e());
}

/** @brief DD-precision log10(x); see ndarray_dd_log10 in src/dd_math.c. */
__device__ inline dd_real dd_log10(dd_real a) {
    return dd_mul(dd_log(a), dd_log10e());
}

/** @brief DD-precision logb(x); see ndarray_dd_logb in src/dd_math.c. */
__device__ inline dd_real dd_logb(dd_real a) {
    return dd_make(logb(a.hi), 0.0);
}

/* ── DD-precision hyperbolic functions (device side) ─────────────────────
   sinh / cosh / tanh / asinh / acosh / atanh evaluated entirely in DD
   arithmetic by composing the DD exp / expm1 / log / log1p / sqrt
   primitives above. Each carries ~32 decimal digits, matching the CPU
   libquadmath path; they replace the former dd → double → libm → dd
   fp64-tier kernels. Algorithms are byte-identical to the CPU twins in
   src/dd_math.c (ndarray_dd_sinh …) so CPU↔GPU parity holds. */

/** @brief DD-precision sinh(x); see ndarray_dd_sinh in src/dd_math.c. */
__device__ inline dd_real dd_sinh(dd_real a) {
    if (isnan(a.hi)) return a;
    /* Near 0 the direct (e^x − e^−x)/2 cancels; use the expm1 identity
       sinh(x) = u·(u+2) / (2·(u+1)) with u = expm1(x). */
    if (a.hi > -0.5 && a.hi < 0.5) {
        dd_real u   = dd_expm1(a);
        dd_real num = dd_mul(u, dd_add(u, dd_make(2.0, 0.0)));
        dd_real den = dd_mul(dd_make(2.0, 0.0), dd_add(u, dd_make(1.0, 0.0)));
        return dd_div(num, den);
    }
    /* |x| ≥ 0.5: no harmful cancellation. exp(±inf) over/underflow is
       handled inside dd_exp, so large |x| yields ±inf with no inf−inf. */
    dd_real ex  = dd_exp(a);
    dd_real enx = dd_exp(dd_neg(a));
    return dd_mul(dd_sub(ex, enx), dd_make(0.5, 0.0));
}

/** @brief DD-precision cosh(x); see ndarray_dd_cosh in src/dd_math.c. */
__device__ inline dd_real dd_cosh(dd_real a) {
    if (isnan(a.hi)) return a;
    /* cosh(x) = (e^x + e^−x)/2 — both terms positive, so no cancellation
       at any magnitude and cosh(0) = 1 stays exact. */
    dd_real ex  = dd_exp(a);
    dd_real enx = dd_exp(dd_neg(a));
    return dd_mul(dd_add(ex, enx), dd_make(0.5, 0.0));
}

/** @brief DD-precision tanh(x); see ndarray_dd_tanh in src/dd_math.c. */
__device__ inline dd_real dd_tanh(dd_real a) {
    if (isnan(a.hi)) return a;
    /* |x| > 40: tanh saturates to ±1 below DD epsilon (1 − tanh(40) ≈
       1e-35) and the expm1(2x) form would overflow — return ±1. This
       also folds in the ±inf inputs. */
    if (a.hi >  40.0) return dd_make( 1.0, 0.0);
    if (a.hi < -40.0) return dd_make(-1.0, 0.0);
    /* tanh(x) = v/(v+2) with v = expm1(2x): no cancellation near 0, no
       overflow for |x| ≤ 40 (e^80 ≈ 5.5e34 ≪ DBL_MAX). */
    dd_real two = dd_make(2.0, 0.0);
    dd_real v   = dd_expm1(dd_mul(a, two));
    return dd_div(v, dd_add(v, two));
}

/** @brief DD-precision asinh(x); see ndarray_dd_arcsinh in src/dd_math.c. */
__device__ inline dd_real dd_asinh(dd_real a) {
    if (isnan(a.hi) || isinf(a.hi)) return a;   /* asinh(±inf) = ±inf */
    int     neg = (a.hi < 0.0);
    dd_real ax  = dd_abs(a);
    dd_real one = dd_make(1.0, 0.0);
    dd_real r;
    if (ax.hi > 1e150) {
        /* x² would overflow; asinh(x) ≈ ln(2|x|) = ln|x| + ln 2. */
        r = dd_add(dd_log(ax), dd_ln2());
    } else {
        /* asinh(x) = log1p(x + x²/(1 + sqrt(1+x²))); the x²/(1+s) term
           equals sqrt(x²+1) − 1 without cancellation, and log1p keeps
           full precision near 0. Computed on |x| then sign-restored so
           the x + sqrt(x²+1) argument never cancels for x < 0. */
        dd_real t = dd_mul(ax, ax);
        dd_real s = dd_sqrt(dd_add(t, one));
        r = dd_log1p(dd_add(ax, dd_div(t, dd_add(one, s))));
    }
    return neg ? dd_neg(r) : r;
}

/** @brief DD-precision acosh(x); see ndarray_dd_arccosh in src/dd_math.c. */
__device__ inline dd_real dd_acosh(dd_real a) {
    if (isnan(a.hi))  return a;
    if (a.hi < 1.0)   return dd_make(nan(""), 0.0);  /* domain: x ≥ 1 */
    if (isinf(a.hi))  return a;                       /* acosh(+inf) = +inf */
    if (a.hi > 1e150) return dd_add(dd_log(a), dd_ln2());  /* ≈ ln(2x) */
    /* acosh(x) = log1p((x−1) + sqrt((x−1)(x+1))); the split sqrt avoids
       cancellation as x → 1 where acosh → 0. */
    dd_real one  = dd_make(1.0, 0.0);
    dd_real w    = dd_sub(a, one);
    dd_real root = dd_mul(dd_sqrt(w), dd_sqrt(dd_add(w, dd_make(2.0, 0.0))));
    return dd_log1p(dd_add(w, root));
}

/** @brief DD-precision atanh(x); see ndarray_dd_arctanh in src/dd_math.c. */
__device__ inline dd_real dd_atanh(dd_real a) {
    if (isnan(a.hi)) return a;
    dd_real ax = dd_abs(a);
    if (ax.hi > 1.0) return dd_make(nan(""), 0.0);    /* domain: |x| ≤ 1 */
    if (ax.hi == 1.0 && a.lo == 0.0)                  /* atanh(±1) = ±inf */
        return dd_make(a.hi > 0.0 ? INFINITY : -INFINITY, 0.0);
    /* atanh(x) = ½·log1p(2x/(1−x)): no cancellation near 0; the |x| = 1
       endpoint is handled above (1−x = 0 would make dd_div yield NaN). */
    dd_real one = dd_make(1.0, 0.0);
    dd_real num = dd_add(a, a);
    dd_real den = dd_sub(one, a);
    return dd_mul(dd_make(0.5, 0.0), dd_log1p(dd_div(num, den)));
}

/**
 * @brief DD strict less-than, NaN-safe.
 *
 * Returns true iff `a < b` under the lexicographic (hi, lo) ordering.
 * NaN in either operand makes both `< ` and `> ` checks false, falling
 * through to a NaN-vs-NaN comparison on `.lo` which is also false — so
 * the predicate returns false whenever either operand is NaN, matching
 * IEEE-754's "unordered" contract for ordered comparisons.
 */
__device__ inline bool dd_lt(dd_real a, dd_real b) {
    if (a.hi < b.hi) return true;
    if (a.hi > b.hi) return false;
    return a.lo < b.lo;
}

/**
 * @brief DD clamp matching PyTorch's clamp semantics.
 *
 * Equivalent to `std::min(std::max(x, lo), hi)` for ordered inputs.
 * NaN propagation:
 *  - NaN in `x` propagates to the result (both `lt(x, lo)` and
 *    `lt(hi, x)` are false, so the original NaN survives both
 *    branches);
 *  - NaN in `lo` or `hi` is swallowed (the corresponding bound check
 *    returns false, so the value survives), matching PyTorch's CPU
 *    kernel behaviour.
 * Also gives the documented PyTorch result when `lo > hi`: the
 * answer is `hi`.
 */
__device__ inline dd_real dd_clip(dd_real x, dd_real lo, dd_real hi) {
    dd_real _y = dd_lt(x, lo)  ? lo : x;     /* max(x, lo)  */
    return       dd_lt(hi, _y) ? hi : _y;    /* min(_y, hi) */
}

/* sinc(π·x) with x stored as DD. For small |x| ≤ 0.1 (|πx| ≤ 0.314) the
   Maclaurin series sinc(πx) = Σ_{k≥0} (-1)^k (πx)^{2k}/(2k+1)! is summed
   in DD arithmetic so the (hi, lo) pair stays meaningful — the previous
   fp64 fallback collapsed `sinc(1e-10)` to exactly `1.0`, losing the
   `~1.6e-21` deviation the CPU (libquadmath sinq) captures.  Twelve
   terms (through (πx)^24/25!) drive the worst-case remainder at the
   |x| = 0.1 boundary below DD epsilon (~1.2e-32); the original 5-term
   cutoff held only ~16 digits there.  Terms are built by the recurrence
   t_k = t_{k-1}·(-(πx)²)/((2k)(2k+1)) so every divisor (≤ 24·25) is
   exactly representable in fp64 — unlike the high factorials 19!/21!/23!
   which are not.  For larger |x| the argument reduction in `sin(πx)` is
   the dominant error term (precision-bound by fp64), so we keep the fp64
   path there — full DD trig would need a CORDIC ladder. */
__device__ inline dd_real dd_sinc(dd_real x) {
    if (x.hi == 0.0 && x.lo == 0.0) return dd_make(1.0, 0.0);

    double abs_hi = (x.hi < 0.0) ? -x.hi : x.hi;
    if (abs_hi <= 0.1) {
        /* π in double-double: hi = nearest-double(π), lo = π - hi. */
        const dd_real pi_dd = dd_make(3.141592653589793,
                                       1.2246467991473532e-16);
        dd_real px      = dd_mul(pi_dd, x);
        dd_real px2     = dd_mul(px, px);
        dd_real neg_px2 = dd_make(-px2.hi, -px2.lo);  /* -(πx)² */

        dd_real term   = dd_make(1.0, 0.0);           /* t_0 = 1 */
        dd_real result = term;
        for (int k = 1; k <= 12; k++) {
            double denom = (double)(2 * k) * (double)(2 * k + 1);
            term   = dd_div(dd_mul(term, neg_px2), dd_make(denom, 0.0));
            result = dd_add(result, term);
        }
        return result;
    }

    /* Larger |x|: argument reduction is the bottleneck; fp64 path. */
    double xd = dd_to_double(x);
    double arg = 3.14159265358979323846 * xd;
    double r = sin(arg) / arg;
    return dd_make(r, 0.0);
}

/* ──────────────────────────────────────────────────────────────────────────
   Templated unary kernels — one body per op-shape.

   Numeric semantics (parallel to CPU):
   - `negate`/`square` on integers wrap modulo 2^N (cast through unsigned).
   - `abs` on signed ints uses the wrap-on-INT_MIN convention NumPy follows.
   - `sign` always returns -1, 0, or +1 (cast to the source dtype).
   - `reciprocal`/`sqrt`/`rsqrt`/`sinc` operate only on floating dtypes; the
     dispatcher promotes integer inputs to a float dtype before launch.
   ────────────────────────────────────────────────────────────────────────── */

/* Integer-template branch: cast through `unsigned` of the same width to
   wrap negation / squaring modulo 2^N (NumPy + PyTorch contract,
   especially for `-INT_MIN` on signed types). */
template <typename T>
__device__ inline typename std::enable_if<std::is_integral<T>::value, T>::type
tcuda_negate_v(T x) {
    typedef typename std::make_unsigned<T>::type UT;
    return (T)(UT)(-(UT)x);
}
template <typename T>
__device__ inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
tcuda_negate_v(T x) { return -x; }

template <typename T>
__device__ inline typename std::enable_if<std::is_integral<T>::value, T>::type
tcuda_square_v(T x) {
    typedef typename std::make_unsigned<T>::type UT;
    return (T)(UT)((UT)x * (UT)x);
}
template <typename T>
__device__ inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
tcuda_square_v(T x) { return x * x; }

/* `abs` is only instantiated for signed integer dtypes (the dispatcher
   short-circuits to a no-op for unsigned). The wrapping cast handles
   `abs(INT_MIN)` symmetrically with negation. */
template <typename T> __device__ inline T tcuda_abs_signed_v(T x) {
    typedef typename std::make_unsigned<T>::type UT;
    return (x < (T)0) ? (T)(UT)(-(UT)x) : x;
}

/* Sign returns -1 / 0 / +1 in the source dtype. The signed branch is
   the canonical three-way subtraction; the unsigned branch avoids the
   always-false `x < 0` comparison NVCC otherwise warns about. */
template <typename T>
__device__ inline typename std::enable_if<std::is_signed<T>::value, T>::type
tcuda_sign_v(T x) { return (T)((x > (T)0) - (x < (T)0)); }
template <typename T>
__device__ inline typename std::enable_if<std::is_unsigned<T>::value, T>::type
tcuda_sign_v(T x) { return (T)(x != (T)0); }


/* In-place kernels for arity-0 ops (single buffer). */
#define TYPED_UNOP_KERNEL_INPLACE(NAME, EXPR)                                       \
template <typename T>                                                               \
__global__ void NAME(T *a, int n) {                                                 \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                                  \
    if (i < n) { T x = a[i]; a[i] = (EXPR); }                                       \
}

TYPED_UNOP_KERNEL_INPLACE(tcuda_negate_kernel,  tcuda_negate_v<T>(x))
TYPED_UNOP_KERNEL_INPLACE(tcuda_abs_int_kernel, tcuda_abs_signed_v<T>(x))
TYPED_UNOP_KERNEL_INPLACE(tcuda_sign_kernel,    tcuda_sign_v<T>(x))
TYPED_UNOP_KERNEL_INPLACE(tcuda_square_kernel,  tcuda_square_v<T>(x))

/* Floating-point unary kernels — fp32 / fp64 share the body but parameterise
   the math functions through a small trait so each instantiation pulls the
   right intrinsic (sqrtf vs sqrt, sinf vs sin). */
template <typename T> __device__ inline T tcuda_sqrt_fp (T x);
template <>           __device__ inline float  tcuda_sqrt_fp<float >(float  x) { return sqrtf(x); }
template <>           __device__ inline double tcuda_sqrt_fp<double>(double x) { return sqrt (x); }

template <typename T> __device__ inline T tcuda_sin_fp (T x);
template <>           __device__ inline float  tcuda_sin_fp<float >(float  x) { return sinf(x); }
template <>           __device__ inline double tcuda_sin_fp<double>(double x) { return sin (x); }

template <typename T> __device__ inline T tcuda_fabs_fp(T x);
template <>           __device__ inline float  tcuda_fabs_fp<float >(float  x) { return fabsf(x); }
template <>           __device__ inline double tcuda_fabs_fp<double>(double x) { return fabs (x); }

/* Transcendental traits — every op routes through the suffixed `f` intrinsic
   for `float` and the unsuffixed CUDA libdevice intrinsic for `double` so
   each instantiation pulls the correct precision without a runtime cast. */
template <typename T> __device__ inline T tcuda_exp_fp  (T x);
template <>           __device__ inline float  tcuda_exp_fp <float >(float  x) { return expf (x); }
template <>           __device__ inline double tcuda_exp_fp <double>(double x) { return exp  (x); }

template <typename T> __device__ inline T tcuda_exp2_fp (T x);
template <>           __device__ inline float  tcuda_exp2_fp<float >(float  x) { return exp2f(x); }
template <>           __device__ inline double tcuda_exp2_fp<double>(double x) { return exp2 (x); }

template <typename T> __device__ inline T tcuda_expm1_fp(T x);
template <>           __device__ inline float  tcuda_expm1_fp<float >(float  x) { return expm1f(x); }
template <>           __device__ inline double tcuda_expm1_fp<double>(double x) { return expm1 (x); }

template <typename T> __device__ inline T tcuda_log_fp  (T x);
template <>           __device__ inline float  tcuda_log_fp <float >(float  x) { return logf (x); }
template <>           __device__ inline double tcuda_log_fp <double>(double x) { return log  (x); }

template <typename T> __device__ inline T tcuda_log1p_fp(T x);
template <>           __device__ inline float  tcuda_log1p_fp<float >(float  x) { return log1pf(x); }
template <>           __device__ inline double tcuda_log1p_fp<double>(double x) { return log1p (x); }

/* CUDA's log2/log2f intrinsics have up to 1 ULP error and round
   log2(2^k) to k - 1 ULP for some powers of 2 — e.g. CUDA's log2(8.0)
   returns 2.9999999999999996 instead of the exact 3.0 that libm
   delivers. The power-of-2 short-circuit recovers the exact integer
   result on GPU (CPU↔GPU parity for that input class) without
   measurably affecting throughput on the general case: `frexp` is a
   single hardware instruction. The check `m == 0.5` is true precisely
   when @p x is an exact power of two; the unbiased exponent is then
   `e - 1` in `log2`'s ranging convention. */
template <typename T> __device__ inline T tcuda_log2_fp (T x);
template <>           __device__ inline float  tcuda_log2_fp<float >(float  x) {
    if (x > 0.0f && isfinite(x)) {
        int e;
        float m = frexpf(x, &e);
        if (m == 0.5f) return (float)(e - 1);
    }
    return log2f(x);
}
template <>           __device__ inline double tcuda_log2_fp<double>(double x) {
    if (x > 0.0 && isfinite(x)) {
        int e;
        double m = frexp(x, &e);
        if (m == 0.5) return (double)(e - 1);
    }
    return log2(x);
}

template <typename T> __device__ inline T tcuda_log10_fp(T x);
template <>           __device__ inline float  tcuda_log10_fp<float >(float  x) { return log10f(x); }
template <>           __device__ inline double tcuda_log10_fp<double>(double x) { return log10 (x); }

template <typename T> __device__ inline T tcuda_logb_fp (T x);
template <>           __device__ inline float  tcuda_logb_fp<float >(float  x) { return logbf(x); }
template <>           __device__ inline double tcuda_logb_fp<double>(double x) { return logb (x); }

/* Trig / hyperbolic / rounding traits — same DRY pattern as the
   exp/log traits above. Each op specialises to the float vs double
   libdevice intrinsic. `tcuda_<op>_fp<T>(x)` is the only entry the
   templated float kernel needs. */
#define DEF_TCUDA_FP_TRAIT(NAME, F_FN, D_FN)                                          \
    template <typename T> __device__ inline T tcuda_##NAME##_fp (T x);                \
    template <>           __device__ inline float  tcuda_##NAME##_fp<float >(float  x) { return F_FN(x); }  \
    template <>           __device__ inline double tcuda_##NAME##_fp<double>(double x) { return D_FN(x); }

/* `tcuda_sin_fp` already exists above as the sinc trait — reused as-is. */
DEF_TCUDA_FP_TRAIT(cos,     cosf,   cos)
DEF_TCUDA_FP_TRAIT(tan,     tanf,   tan)
DEF_TCUDA_FP_TRAIT(arcsin,  asinf,  asin)
DEF_TCUDA_FP_TRAIT(arccos,  acosf,  acos)
DEF_TCUDA_FP_TRAIT(arctan,  atanf,  atan)
DEF_TCUDA_FP_TRAIT(sinh,    sinhf,  sinh)
DEF_TCUDA_FP_TRAIT(cosh,    coshf,  cosh)
DEF_TCUDA_FP_TRAIT(tanh,    tanhf,  tanh)
DEF_TCUDA_FP_TRAIT(arcsinh, asinhf, asinh)
DEF_TCUDA_FP_TRAIT(arccosh, acoshf, acosh)
DEF_TCUDA_FP_TRAIT(arctanh, atanhf, atanh)
DEF_TCUDA_FP_TRAIT(rint,    rintf,  rint)
DEF_TCUDA_FP_TRAIT(trunc,   truncf, trunc)
DEF_TCUDA_FP_TRAIT(floor,   floorf, floor)
DEF_TCUDA_FP_TRAIT(ceil,    ceilf,  ceil)

#undef DEF_TCUDA_FP_TRAIT

/* Degrees / radians use a multiplicative constant rather than a libm
   intrinsic. Inline the constant per type so the launch picks the
   right precision (fp32 constant for `float`, fp64 for `double`). */
template <typename T> __device__ inline T tcuda_deg_factor();
template <>           __device__ inline float  tcuda_deg_factor<float >() { return (float)(180.0 / 3.14159265358979323846); }
template <>           __device__ inline double tcuda_deg_factor<double>() { return 180.0 / 3.14159265358979323846; }
template <typename T> __device__ inline T tcuda_rad_factor();
template <>           __device__ inline float  tcuda_rad_factor<float >() { return (float)(3.14159265358979323846 / 180.0); }
template <>           __device__ inline double tcuda_rad_factor<double>() { return 3.14159265358979323846 / 180.0; }

template <typename T>
__global__ void tcuda_abs_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_fabs_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_recip_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = (T)1 / a[i];
}
template <typename T>
__global__ void tcuda_sqrt_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_sqrt_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_rsqrt_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = (T)1 / tcuda_sqrt_fp<T>(a[i]);
}

/* Transcendental float kernels — one template per op so the dispatcher
   can instantiate against either `float` or `double` without an
   intermediate cast. The libdevice intrinsics produced by `tcuda_<op>_fp`
   round to the IEEE-754 nearest-even mode that matches the libm path
   on the CPU. */
template <typename T>
__global__ void tcuda_exp_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_exp_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_exp2_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_exp2_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_expm1_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_expm1_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_log_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_log_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_log1p_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_log1p_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_log2_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_log2_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_log10_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_log10_fp<T>(a[i]);
}
template <typename T>
__global__ void tcuda_logb_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = tcuda_logb_fp<T>(a[i]);
}

/* Trig / hyperbolic / rounding templated float kernels. Each kernel
   dispatches to its dtype-specialised libdevice intrinsic via the
   matching `tcuda_<op>_fp<T>` trait. The macro keeps the 16 kernels
   shaped identically. */
#define DEF_TCUDA_FLOAT_UNOP(NAME, TRAIT)                                             \
    template <typename T>                                                             \
    __global__ void tcuda_##NAME##_float_kernel(T *a, int n) {                        \
        int i = threadIdx.x + blockIdx.x * blockDim.x;                                \
        if (i < n) a[i] = TRAIT<T>(a[i]);                                             \
    }

DEF_TCUDA_FLOAT_UNOP(sin,     tcuda_sin_fp)
DEF_TCUDA_FLOAT_UNOP(cos,     tcuda_cos_fp)
DEF_TCUDA_FLOAT_UNOP(tan,     tcuda_tan_fp)
DEF_TCUDA_FLOAT_UNOP(arcsin,  tcuda_arcsin_fp)
DEF_TCUDA_FLOAT_UNOP(arccos,  tcuda_arccos_fp)
DEF_TCUDA_FLOAT_UNOP(arctan,  tcuda_arctan_fp)
DEF_TCUDA_FLOAT_UNOP(sinh,    tcuda_sinh_fp)
DEF_TCUDA_FLOAT_UNOP(cosh,    tcuda_cosh_fp)
DEF_TCUDA_FLOAT_UNOP(tanh,    tcuda_tanh_fp)
DEF_TCUDA_FLOAT_UNOP(arcsinh, tcuda_arcsinh_fp)
DEF_TCUDA_FLOAT_UNOP(arccosh, tcuda_arccosh_fp)
DEF_TCUDA_FLOAT_UNOP(arctanh, tcuda_arctanh_fp)
DEF_TCUDA_FLOAT_UNOP(rint,    tcuda_rint_fp)
DEF_TCUDA_FLOAT_UNOP(trunc,   tcuda_trunc_fp)
DEF_TCUDA_FLOAT_UNOP(floor,   tcuda_floor_fp)
DEF_TCUDA_FLOAT_UNOP(ceil,    tcuda_ceil_fp)

#undef DEF_TCUDA_FLOAT_UNOP

/* Degrees / radians kernels — linear multiplication, no libm. */
template <typename T>
__global__ void tcuda_degrees_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = a[i] * tcuda_deg_factor<T>();
}
template <typename T>
__global__ void tcuda_radians_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = a[i] * tcuda_rad_factor<T>();
}
template <typename T>
__global__ void tcuda_sinc_float_kernel(T *a, int n) {
    /* Normalised sinc(x) = sin(π·x) / (π·x), with sinc(0) = 1. */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        T x  = a[i];
        if (x == (T)0) { a[i] = (T)1; }
        else {
            T px = (T)3.14159265358979323846 * x;
            a[i] = tcuda_sin_fp<T>(px) / px;
        }
    }
}
template <typename T>
__global__ void tcuda_sign_float_kernel(T *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        T x = a[i];
        /* PyTorch sign(NaN) = NaN. The IEEE-754 self-compare `x != x` is
           the canonical NaN test and works for any float dtype. */
        if      (x != x)   a[i] = x;
        else if (x > (T)0) a[i] = (T) 1;
        else if (x < (T)0) a[i] = (T)-1;
        else               a[i] = (T) 0;
    }
}
template <typename T>
__global__ void tcuda_clip_kernel(T *a, T lo, T hi, int n) {
    /* For integer T this is plain min/max via cppref-style branching.
       For float T the (a < b) ? b : a / (b < a) ? b : a pair matches
       std::max / std::min — see UNARY_FLOAT_BODY for the NaN rationale. */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        T x  = a[i];
        T _y = (x < lo) ? lo : x;
        a[i] = (hi < _y) ? hi : _y;
    }
}

/* __half (float16, stored as uint16_t) unary kernels — compute through float
   for accuracy, matching the binary path. */
__global__ void tcuda_negate_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = __float2half(-__half2float(a[i]));
}
__global__ void tcuda_abs_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = __float2half(fabsf(__half2float(a[i])));
}
__global__ void tcuda_sign_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float x = __half2float(a[i]);
        /* PyTorch sign(NaN) = NaN. NaN-aware branch before the
           branchless triplet, then `__float2half(NaN)` round-trips
           the NaN through the half encoding. */
        float r;
        if      (x != x)   r = x;
        else if (x > 0.0f) r =  1.0f;
        else if (x < 0.0f) r = -1.0f;
        else               r =  0.0f;
        a[i] = __float2half(r);
    }
}
__global__ void tcuda_recip_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = __float2half(1.0f / __half2float(a[i]));
}
__global__ void tcuda_sqrt_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = __float2half(sqrtf(__half2float(a[i])));
}
__global__ void tcuda_rsqrt_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] = __float2half(1.0f / sqrtf(__half2float(a[i])));
}
__global__ void tcuda_square_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float x = __half2float(a[i]);
        a[i] = __float2half(x * x);
    }
}
__global__ void tcuda_clip_half_kernel(__half *a, float lo, float hi, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float x  = __half2float(a[i]);
        float _y = (x < lo) ? lo : x;
        float r  = (hi < _y) ? hi : _y;
        a[i] = __float2half(r);
    }
}
__global__ void tcuda_sinc_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float x = __half2float(a[i]);
        float r;
        if (x == 0.0f) r = 1.0f;
        else { float px = 3.14159265358979323846f * x; r = sinf(px) / px; }
        a[i] = __float2half(r);
    }
}

/* Transcendental fp16 kernels — round-trip through float for accuracy.
   The native half intrinsics (`hexp`, `hlog`, …) round once to the
   nearest representable half value, which can diverge from the CPU's
   `double → fp16` path on edge inputs. Routing through `float` keeps
   bit-for-bit parity with the CPU loop that test 108 enforces. The
   layout mirrors the abs/sqrt/recip/sinc unary half-kernels above. */
#define HALF_UNOP_KERNEL(NAME, EXPR_F)                                              \
__global__ void NAME(__half *a, int n) {                                            \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                                  \
    if (i < n) a[i] = __float2half((EXPR_F)(__half2float(a[i])));                   \
}
HALF_UNOP_KERNEL(tcuda_exp_half_kernel,   expf)
HALF_UNOP_KERNEL(tcuda_exp2_half_kernel,  exp2f)
HALF_UNOP_KERNEL(tcuda_expm1_half_kernel, expm1f)
HALF_UNOP_KERNEL(tcuda_log_half_kernel,   logf)
HALF_UNOP_KERNEL(tcuda_log1p_half_kernel, log1pf)
HALF_UNOP_KERNEL(tcuda_log2_half_kernel,  tcuda_log2_fp<float>)
HALF_UNOP_KERNEL(tcuda_log10_half_kernel, log10f)
HALF_UNOP_KERNEL(tcuda_logb_half_kernel,  logbf)
/* Trig / hyperbolic / rounding fp16 kernels — round-trip through
   float32 (same precision contract as the exp/log half family). */
HALF_UNOP_KERNEL(tcuda_sin_half_kernel,      sinf)
HALF_UNOP_KERNEL(tcuda_cos_half_kernel,      cosf)
HALF_UNOP_KERNEL(tcuda_tan_half_kernel,      tanf)
HALF_UNOP_KERNEL(tcuda_arcsin_half_kernel,   asinf)
HALF_UNOP_KERNEL(tcuda_arccos_half_kernel,   acosf)
HALF_UNOP_KERNEL(tcuda_arctan_half_kernel,   atanf)
HALF_UNOP_KERNEL(tcuda_sinh_half_kernel,     sinhf)
HALF_UNOP_KERNEL(tcuda_cosh_half_kernel,     coshf)
HALF_UNOP_KERNEL(tcuda_tanh_half_kernel,     tanhf)
HALF_UNOP_KERNEL(tcuda_arcsinh_half_kernel,  asinhf)
HALF_UNOP_KERNEL(tcuda_arccosh_half_kernel,  acoshf)
HALF_UNOP_KERNEL(tcuda_arctanh_half_kernel,  atanhf)
HALF_UNOP_KERNEL(tcuda_rint_half_kernel,     rintf)
HALF_UNOP_KERNEL(tcuda_trunc_half_kernel,    truncf)
HALF_UNOP_KERNEL(tcuda_floor_half_kernel,    floorf)
HALF_UNOP_KERNEL(tcuda_ceil_half_kernel,     ceilf)
#undef HALF_UNOP_KERNEL

/* Degrees / radians fp16 kernels — linear scale via float32. */
__global__ void tcuda_degrees_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float v = __half2float(a[i]) * (float)(180.0 / 3.14159265358979323846);
        a[i] = __float2half(v);
    }
}
__global__ void tcuda_radians_half_kernel(__half *a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float v = __half2float(a[i]) * (float)(3.14159265358979323846 / 180.0);
        a[i] = __float2half(v);
    }
}

/* DD (float128 emulation) unary kernels. Buffer is laid out as
   `(hi[0], lo[0], hi[1], lo[1], …)` so each element occupies 2 doubles. */
#define DD_UNOP_KERNEL(NAME, EXPR)                                                  \
__global__ void NAME(double *a, int n) {                                            \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                                  \
    if (i < n) {                                                                    \
        dd_real x = dd_make(a[2*i], a[2*i+1]);                                      \
        dd_real r = (EXPR);                                                         \
        a[2*i]   = r.hi;                                                            \
        a[2*i+1] = r.lo;                                                            \
    }                                                                               \
}
DD_UNOP_KERNEL(tcuda_negate_dd_kernel, dd_neg(x))
DD_UNOP_KERNEL(tcuda_abs_dd_kernel,    dd_abs(x))
DD_UNOP_KERNEL(tcuda_sign_dd_kernel,   dd_sign(x))
DD_UNOP_KERNEL(tcuda_recip_dd_kernel,  dd_recip(x))
DD_UNOP_KERNEL(tcuda_sqrt_dd_kernel,   dd_sqrt(x))
DD_UNOP_KERNEL(tcuda_rsqrt_dd_kernel,  dd_rsqrt(x))
DD_UNOP_KERNEL(tcuda_square_dd_kernel, dd_mul(x, x))
DD_UNOP_KERNEL(tcuda_sinc_dd_kernel,   dd_sinc(x))

/* Transcendental DD kernels — each op runs entirely in DD arithmetic
   (range reduction + Horner-evaluated DD Taylor / atanh series), so
   the (hi, lo) pair carries ~30 decimal digits of precision matching
   the CPU libquadmath path. The previous fp64 fallback (`dd_make(exp
   (dd_to_double(x)), 0.0)`) collapsed every result to ~15 digits;
   the new path uses `dd_exp` / `dd_log` / etc. defined above. */
DD_UNOP_KERNEL(tcuda_exp_dd_kernel,    dd_exp   (x))
DD_UNOP_KERNEL(tcuda_exp2_dd_kernel,   dd_exp2  (x))
DD_UNOP_KERNEL(tcuda_expm1_dd_kernel,  dd_expm1 (x))
DD_UNOP_KERNEL(tcuda_log_dd_kernel,    dd_log   (x))
DD_UNOP_KERNEL(tcuda_log1p_dd_kernel,  dd_log1p (x))
DD_UNOP_KERNEL(tcuda_log2_dd_kernel,   dd_log2  (x))
DD_UNOP_KERNEL(tcuda_log10_dd_kernel,  dd_log10 (x))
DD_UNOP_KERNEL(tcuda_logb_dd_kernel,   dd_logb  (x))

/* Trig DD kernels — still fp64 precision tier (dd → double → libm → dd).
   Full 113-bit trig needs argument reduction by a DD π and remains on the
   libquadmath CPU path; the GPU dd trig tops out at fp64. */
DD_UNOP_KERNEL(tcuda_sin_dd_kernel,      dd_make(sin   (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_cos_dd_kernel,      dd_make(cos   (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_tan_dd_kernel,      dd_make(tan   (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_arcsin_dd_kernel,   dd_make(asin  (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_arccos_dd_kernel,   dd_make(acos  (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_arctan_dd_kernel,   dd_make(atan  (dd_to_double(x)), 0.0))

/* Hyperbolic DD kernels — full DD precision (~32 digits) via the dd_sinh …
   dd_atanh helpers above, which compose the DD exp/log/sqrt primitives.
   They previously degraded to fp64 (dd_make(sinh(dd_to_double(x)), 0.0));
   the dd path now matches the CPU libquadmath result to ~31–32 digits. */
DD_UNOP_KERNEL(tcuda_sinh_dd_kernel,     dd_sinh  (x))
DD_UNOP_KERNEL(tcuda_cosh_dd_kernel,     dd_cosh  (x))
DD_UNOP_KERNEL(tcuda_tanh_dd_kernel,     dd_tanh  (x))
DD_UNOP_KERNEL(tcuda_arcsinh_dd_kernel,  dd_asinh (x))
DD_UNOP_KERNEL(tcuda_arccosh_dd_kernel,  dd_acosh (x))
DD_UNOP_KERNEL(tcuda_arctanh_dd_kernel,  dd_atanh (x))
/* Rounding ops preserve the full fp128 input bits — only the fp64 hi
   word changes, so we keep the original lo word. dd_rint/trunc/floor/
   ceil on a pure-double input value-equals libm rounding of the hi
   word. */
DD_UNOP_KERNEL(tcuda_rint_dd_kernel,     dd_make(rint  (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_trunc_dd_kernel,    dd_make(trunc (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_floor_dd_kernel,    dd_make(floor (dd_to_double(x)), 0.0))
DD_UNOP_KERNEL(tcuda_ceil_dd_kernel,     dd_make(ceil  (dd_to_double(x)), 0.0))

/* Degrees / radians DD kernels — fp64-precision multiplicative scale. */
DD_UNOP_KERNEL(tcuda_degrees_dd_kernel,
               dd_make(dd_to_double(x) * (180.0 / 3.14159265358979323846), 0.0))
DD_UNOP_KERNEL(tcuda_radians_dd_kernel,
               dd_make(dd_to_double(x) * (3.14159265358979323846 / 180.0), 0.0))

__global__ void tcuda_clip_dd_kernel(double *a, double lo_hi, double lo_lo,
                                     double hi_hi, double hi_lo, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        dd_real x  = dd_make(a[2*i], a[2*i+1]);
        dd_real lo = dd_make(lo_hi, lo_lo);
        dd_real hi = dd_make(hi_hi, hi_lo);
        dd_real r  = dd_clip(x, lo, hi);
        a[2*i]   = r.hi;
        a[2*i+1] = r.lo;
    }
}

/* ──────────────────────────────────────────────────────────────────────────
   float4 (E2M1) and float8 (E4M3) GPU casts.

   float4 has only 16 distinct values; we cast through float (lossless: all
   representable values are exact in fp32). The encoder picks the nearest
   representable nibble.

   float8 (E4M3) has range [-240, 240] with ~3 mantissa bits; all values fit
   exactly in float16 (mantissa 10, exponent 5) so casting fp8 ⇄ float16 is
   lossless. We round-trip through float for the math, which gives the same
   result as float16 with more headroom.
   ────────────────────────────────────────────────────────────────────────── */

__constant__ float c_fp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ inline float dev_fp4_to_float(uint8_t nibble) {
    return c_fp4_lut[nibble & 0x0F];
}

__device__ inline uint8_t dev_float_to_fp4(float val) {
    /* Find nearest representable. Branchless tournament. */
    uint8_t best = 0;
    float best_err = fabsf(val - c_fp4_lut[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float err = fabsf(val - c_fp4_lut[i]);
        if (err < best_err) { best_err = err; best = (uint8_t)i; }
    }
    return best;
}

__device__ inline float dev_fp8_to_float(uint8_t fp8) {
    int sign = (fp8 >> 7) & 1;
    int exp  = (fp8 >> 3) & 0xF;
    int man  = fp8 & 0x7;
    if (exp == 0xF) return nanf("");
    float val;
    if (exp == 0) {
        val = ((float)man / 8.0f) * ldexpf(1.0f, -6);
    } else {
        val = (1.0f + (float)man / 8.0f) * ldexpf(1.0f, exp - 7);
    }
    return sign ? -val : val;
}

__device__ inline uint8_t dev_float_to_fp8(float val) {
    if (isnan(val)) return 0xFF;
    int sign = (val < 0.0f || (val == 0.0f && (1.0f / val) < 0)) ? 1 : 0;
    float ax = fabsf(val);
    if (ax == 0.0f) return (uint8_t)(sign << 7);
    if (ax > 240.0f) ax = 240.0f;

    int e;
    float frac = frexpf(ax, &e);   /* ax = frac * 2^e, frac in [0.5, 1.0) */
    int biased_exp = e + 6;

    if (biased_exp <= 0) {
        int man = (int)(ax * 512.0f + 0.5f);
        if (man > 7) man = 7;
        return (uint8_t)((sign << 7) | man);
    }
    if (biased_exp >= 15) {
        return (uint8_t)((sign << 7) | (14 << 3) | 7);
    }
    int man = (int)((2.0f * frac - 1.0f) * 8.0f + 0.5f);
    if (man > 7) { man = 0; biased_exp++; }
    if (biased_exp >= 15) {
        return (uint8_t)((sign << 7) | (14 << 3) | 7);
    }
    return (uint8_t)((sign << 7) | ((biased_exp & 0xF) << 3) | (man & 0x7));
}

/* Cast kernels: each thread converts one element. */
__global__ void tcuda_cast_fp4_to_f32_kernel(const uint8_t *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = dev_fp4_to_float(src[i]);
}
__global__ void tcuda_cast_f32_to_fp4_kernel(const float *src, uint8_t *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = dev_float_to_fp4(src[i]);
}
__global__ void tcuda_cast_fp4_to_f16_kernel(const uint8_t *src, __half *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(dev_fp4_to_float(src[i]));
}
__global__ void tcuda_cast_f16_to_fp4_kernel(const __half *src, uint8_t *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = dev_float_to_fp4(__half2float(src[i]));
}

__global__ void tcuda_cast_fp8_to_f32_kernel(const uint8_t *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = dev_fp8_to_float(src[i]);
}
__global__ void tcuda_cast_f32_to_fp8_kernel(const float *src, uint8_t *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = dev_float_to_fp8(src[i]);
}
__global__ void tcuda_cast_fp8_to_f16_kernel(const uint8_t *src, __half *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(dev_fp8_to_float(src[i]));
}
__global__ void tcuda_cast_f16_to_fp8_kernel(const __half *src, uint8_t *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = dev_float_to_fp8(__half2float(src[i]));
}

/* ── Typed reduction kernels (two-pass shared-memory tree) ─────────────────
 *
 * Layout:
 *   Pass 1 (per-block tree reduce on shared memory):
 *     - Phase 1a: each thread accumulates its slice of input into a private
 *       `double` via a grid-stride loop. With a capped grid of
 *       REDUCE_MAX_BLOCKS we keep the cross-block atomic count bounded
 *       regardless of n.
 *     - Phase 1b: per-thread partials land in shared memory and the block
 *       collapses them via a power-of-two tree reduction. Every step calls
 *       __syncthreads(), so all threads in the block participate at the
 *       same control-flow point — no warp shuffles, no divergent shuffle
 *       UB on Volta+.
 *   Pass 2 (cross-block merge):
 *     - Thread 0 of each block atomically merges the block's partial into
 *       the global accumulator slot via the per-op atomic CAS helper.
 *
 * Total atomic ops = min(num_blocks, REDUCE_MAX_BLOCKS), independent of n.
 * The single-pass (one-atomic-per-element) version this replaced was
 * correct on any architecture but did O(n) atomicCAS spins, which on
 * Ampere with n in the millions could be 100× slower than this tree.
 *
 * Caller contract (in cuda_math.h):
 *   - vmalloc a single `double` slot on GPU
 *   - cudaMemcpy the operation's identity into it (0/1/+DBL_MAX/-DBL_MAX)
 *   - call the typed wrapper
 *   - cudaMemcpy the slot back to host, vfree it.
 */
#define REDUCE_BLOCK       256
#define REDUCE_MAX_BLOCKS  1024

/* ── Reduction kernel body macros ──────────────────────────────────────────
 * The four operations differ only in identity value, the binary combine
 * applied to two doubles, and the cross-block atomic. READ_EXPR is the
 * dtype-specific load (e.g. `(double)a[i]` for native dtypes,
 * `(double)__half2float(a[i])` for fp16, `a[2*i] + a[2*i+1]` for the dd
 * fp128 layout). Each `REDUCE_*_BODY` macro emits the full kernel body
 * for one operation; emitting them as one block ensures every kernel has
 * identical sync placement and tree-reduction structure, eliminating
 * copy-paste drift across 20 specialised kernels. */
#define REDUCE_SUM_COMBINE(a, b)   ((a) + (b))
#define REDUCE_PROD_COMBINE(a, b)  ((a) * (b))
#define REDUCE_MIN_COMBINE(a, b)   prop_fmin((a), (b))
#define REDUCE_MAX_COMBINE(a, b)   prop_fmax((a), (b))

#define REDUCE_KERNEL_BODY(IDENTITY, COMBINE, FINAL_ATOMIC, READ_EXPR)               \
    __shared__ double sdata[REDUCE_BLOCK];                                           \
    int tid = threadIdx.x;                                                           \
    int stride = blockDim.x * gridDim.x;                                             \
    double val = (IDENTITY);                                                         \
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {                \
        val = COMBINE(val, (READ_EXPR));                                             \
    }                                                                                \
    sdata[tid] = val;                                                                \
    __syncthreads();                                                                 \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) sdata[tid] = COMBINE(sdata[tid], sdata[tid + s]);               \
        __syncthreads();                                                             \
    }                                                                                \
    if (tid == 0) FINAL_ATOMIC(result, sdata[0]);

/* Convenience wrappers around REDUCE_KERNEL_BODY for each operation —
   the caller picks dtype and READ_EXPR; identity / combine / atomic
   are wired in automatically. */
#define REDUCE_SUM_BODY(READ_EXPR)                                                   \
    REDUCE_KERNEL_BODY(0.0,      REDUCE_SUM_COMBINE,  atomicAddDouble, READ_EXPR)
#define REDUCE_PROD_BODY(READ_EXPR)                                                  \
    REDUCE_KERNEL_BODY(1.0,      REDUCE_PROD_COMBINE, atomicMulDouble, READ_EXPR)
#define REDUCE_MIN_BODY(READ_EXPR)                                                   \
    REDUCE_KERNEL_BODY( DBL_MAX, REDUCE_MIN_COMBINE,  atomicMinDouble, READ_EXPR)
#define REDUCE_MAX_BODY(READ_EXPR)                                                   \
    REDUCE_KERNEL_BODY(-DBL_MAX, REDUCE_MAX_COMBINE,  atomicMaxDouble, READ_EXPR)

/* Native-dtype templates: any T that implicitly converts to double works
   (all eight integer sizes plus float / double). The READ_EXPR is a
   plain index, which template instantiation specialises per T. */
template <typename T> __global__ void
tcuda_reduce_sum_kernel(const T *a, double *result, int n)  { REDUCE_SUM_BODY ((double)a[i]) }
template <typename T> __global__ void
tcuda_reduce_prod_kernel(const T *a, double *result, int n) { REDUCE_PROD_BODY((double)a[i]) }
template <typename T> __global__ void
tcuda_reduce_min_kernel(const T *a, double *result, int n)  { REDUCE_MIN_BODY ((double)a[i]) }
template <typename T> __global__ void
tcuda_reduce_max_kernel(const T *a, double *result, int n)  { REDUCE_MAX_BODY ((double)a[i]) }

/* float16: read via __half2float so the (double) widening goes through
   the defined IEEE half→single→double chain. */
__global__ void tcuda_reduce_sum_half_kernel (const __half *a, double *result, int n) { REDUCE_SUM_BODY ((double)__half2float(a[i])) }
__global__ void tcuda_reduce_prod_half_kernel(const __half *a, double *result, int n) { REDUCE_PROD_BODY((double)__half2float(a[i])) }
__global__ void tcuda_reduce_min_half_kernel (const __half *a, double *result, int n) { REDUCE_MIN_BODY ((double)__half2float(a[i])) }
__global__ void tcuda_reduce_max_half_kernel (const __half *a, double *result, int n) { REDUCE_MAX_BODY ((double)__half2float(a[i])) }

/* float4 / float8: one byte per element, decoded via dev_fp{4,8}_to_float. */
__global__ void tcuda_reduce_sum_fp4_kernel (const uint8_t *a, double *result, int n) { REDUCE_SUM_BODY ((double)dev_fp4_to_float(a[i])) }
__global__ void tcuda_reduce_prod_fp4_kernel(const uint8_t *a, double *result, int n) { REDUCE_PROD_BODY((double)dev_fp4_to_float(a[i])) }
__global__ void tcuda_reduce_min_fp4_kernel (const uint8_t *a, double *result, int n) { REDUCE_MIN_BODY ((double)dev_fp4_to_float(a[i])) }
__global__ void tcuda_reduce_max_fp4_kernel (const uint8_t *a, double *result, int n) { REDUCE_MAX_BODY ((double)dev_fp4_to_float(a[i])) }
__global__ void tcuda_reduce_sum_fp8_kernel (const uint8_t *a, double *result, int n) { REDUCE_SUM_BODY ((double)dev_fp8_to_float(a[i])) }
__global__ void tcuda_reduce_prod_fp8_kernel(const uint8_t *a, double *result, int n) { REDUCE_PROD_BODY((double)dev_fp8_to_float(a[i])) }
__global__ void tcuda_reduce_min_fp8_kernel (const uint8_t *a, double *result, int n) { REDUCE_MIN_BODY ((double)dev_fp8_to_float(a[i])) }
__global__ void tcuda_reduce_max_fp8_kernel (const uint8_t *a, double *result, int n) { REDUCE_MAX_BODY ((double)dev_fp8_to_float(a[i])) }

/* float128 stored on GPU as interleaved double-double pairs (hi, lo).
   Each element is read as (a[2i] + a[2i+1]) — same precision-flattening
   conversion the CPU side uses in ndarray_fp128_to_double. */
__global__ void tcuda_reduce_sum_dd_kernel (const double *a, double *result, int n) { REDUCE_SUM_BODY (a[2*i] + a[2*i + 1]) }
__global__ void tcuda_reduce_prod_dd_kernel(const double *a, double *result, int n) { REDUCE_PROD_BODY(a[2*i] + a[2*i + 1]) }
__global__ void tcuda_reduce_min_dd_kernel (const double *a, double *result, int n) { REDUCE_MIN_BODY (a[2*i] + a[2*i + 1]) }
__global__ void tcuda_reduce_max_dd_kernel (const double *a, double *result, int n) { REDUCE_MAX_BODY (a[2*i] + a[2*i + 1]) }

/* Byte-wise gather kernel: dst[i] = src[src_offsets[i]] (each element is
   elsize bytes). The host computes src_offsets per broadcast rules and ships
   them via a small int buffer. Keeps GPU code dtype-agnostic. */
__global__ void tcuda_broadcast_by_offsets(const char *src, char *dst,
                                           const int *src_offsets, int n_out,
                                           int elsize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_out) return;
    const char *s = src + (long long)src_offsets[idx] * elsize;
    char *d       = dst + (long long)idx             * elsize;
    for (int b = 0; b < elsize; b++) d[b] = s[b];
}

/* ──────────────────────────────────────────────────────────────────────────
   Strided copy kernel: writes a contiguous, row-major NDArray from a view
   over `src` whose layout is described by (ndim, shape[], strides_b[]).

   Why this kernel exists: a per-row cudaMemcpy loop over a strided slice
   (e.g. column extraction from a 256×256 matrix) pays ~1-2 µs of driver
   latency per row, dominating actual byte movement. A single kernel launch
   amortises the overhead; each thread copies one elsize-byte element after
   decomposing its linear id into the multi-index and computing the source
   offset via strides — signed, so negative-step slices like
   `arr->slice([N-1, 0, -1])` work without a second code path on the host.

   STRIDED_COPY_MAX_NDIM = 16 caps the on-kernel-arg dim count. Real NDArray
   workloads are well under that (most code uses 1-4 dims); higher-ndim
   slices fall back to the per-row cudaMemcpy path in NDArray_Slice. The cap
   keeps the kernel parameter payload at ~200 bytes — well inside the 4 KB
   CUDA kernel-arg budget on sm_70+. */
#define STRIDED_COPY_MAX_NDIM 16

struct StridedCopyDims {
    long long strides_b[STRIDED_COPY_MAX_NDIM]; /* in bytes, signed */
    int       shape    [STRIDED_COPY_MAX_NDIM];
    int       ndim;
    int       elsize;
};

__global__ void tcuda_strided_copy_kernel(
    char       * __restrict__ dst,
    const char * __restrict__ src,
    long long n_elems,
    struct StridedCopyDims dims)
{
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= n_elems) return;

    /* Row-major decomposition: rightmost axis varies fastest. shape[i] is
       always ≥ 1 by construction (zero-element slices short-circuit on the
       host). */
    long long rem     = tid;
    long long src_off = 0;
    for (int i = dims.ndim - 1; i >= 0; i--) {
        long long s   = dims.shape[i];
        long long idx = rem % s;
        rem          /= s;
        src_off      += idx * dims.strides_b[i];
    }

    char       *d = dst + tid * dims.elsize;
    const char *s = src + src_off;

    /* Fast paths for the common dtype sizes; falls back to a byte loop for
       anything exotic. The strides we receive are always multiples of elsize
       (a slice can't break a dtype's internal alignment), so these aligned
       loads/stores are safe. */
    switch (dims.elsize) {
        case 1:  d[0] = s[0]; break;
        case 2:  *(uint16_t *)d = *(const uint16_t *)s; break;
        case 4:  *(uint32_t *)d = *(const uint32_t *)s; break;
        case 8:  *(uint64_t *)d = *(const uint64_t *)s; break;
        case 16: {
            uint64_t       *pd = (uint64_t *)d;
            const uint64_t *ps = (const uint64_t *)s;
            pd[0] = ps[0];
            pd[1] = ps[1];
            break;
        }
        default:
            for (int b = 0; b < dims.elsize; b++) d[b] = s[b];
    }
}

extern "C" {

    int
    cuda_det_float(float *a, float *result, int n) {
        int N = n;
        float *d_A = a;
        cusolverDnHandle_t cusolverH = NULL;
        cudaStream_t stream = NULL;
        cublasHandle_t cublasH = NULL;
        int *d_Ipiv = NULL;
        int *d_info = NULL;
        float *d_U = NULL;
        float *d_work = NULL;
        int *h_Ipiv = NULL;
        int lwork = 0;
        int rc = 1;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

        cusolverDnCreate(&cusolverH);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnSetStream(cusolverH, stream);
        cublasCreate(&cublasH);
        cublasSetStream(cublasH, stream);

        vmalloc((void**)&d_Ipiv, N*sizeof(int));
        vmalloc((void**)&d_info, sizeof(int));
        vmalloc((void**)&d_U, N*N*sizeof(float));

        // copy A to U as cusolverDnSgetrf works in place
        cudaMemcpy(d_U, d_A, N*N*sizeof(float), cudaMemcpyDeviceToDevice);

        /* cusolverDnSgetrf REQUIRES a workspace buffer — the original code
           passed NULL here, which made the LU decomposition produce garbage
           and corrupt subsequent device state (illegal memory access on the
           next vmalloc). Compute the needed size first, then allocate. */
        cusolver_status = cusolverDnSgetrf_bufferSize(cusolverH, N, N, d_U, N, &lwork);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_det_float: cusolverDnSgetrf_bufferSize failed (status=%d)\n",
                    (int)cusolver_status);
            rc = -1;
            goto cleanup;
        }
        if (lwork > 0) {
            vmalloc((void**)&d_work, (size_t)lwork * sizeof(float));
        }

        // LU decompose
        cusolver_status = cusolverDnSgetrf(cusolverH, N, N, d_U, N, d_work, d_Ipiv, d_info);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_det_float: LU decomposition failed (status=%d)\n",
                    (int)cusolver_status);
            rc = -1;
            goto cleanup;
        }

        // Find determinant by product of diagonal elements
        {
            float det = 1.0f;
            int overflow = 0;
            for (int i = 0; i < N; i++) {
                float elem;
                cudaMemcpy(&elem, d_U + i * N + i, sizeof(float), cudaMemcpyDeviceToHost);
                if (fabsf(elem) > FLT_MAX / fabsf(det)) {
                    fprintf(stderr, "cuda_det_float: overflow detected\n");
                    rc = -1;
                    overflow = 1;
                    break;
                }
                if (!isnan(elem) && !isinf(elem)) {
                    det *= elem;
                }
            }
            if (overflow) goto cleanup;

            // Analyze pivot array to calculate number of permutations
            h_Ipiv = new int[N];
            cudaMemcpy(h_Ipiv, d_Ipiv, N*sizeof(int), cudaMemcpyDeviceToHost);

            int numPermutations = 0;
            for(int i = 0; i < N; i++) {
                if(i+1 != h_Ipiv[i]) numPermutations++;
            }

            if(numPermutations % 2 != 0) det = -det;

            cudaMemcpy(result, &det, sizeof(float), cudaMemcpyHostToDevice);
        }

    cleanup:
        if (h_Ipiv) delete[] h_Ipiv;
        if (d_work)  vfree(d_work);
        if (d_U)     vfree(d_U);
        if (d_Ipiv)  vfree(d_Ipiv);
        if (d_info)  vfree(d_info);
        if (cublasH) cublasDestroy(cublasH);
        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (stream)  cudaStreamDestroy(stream);
        return rc;
    }

    void cuda_fill_float(float *a, float value, int n) {
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        fill_float_kernel<<<gridSize, blockSize>>>(a, n, value);
    }

    void cuda_fill_double(double *a, double value, int n) {
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        fill_float_kernel_double<<<gridSize, blockSize>>>(a, n, value);
    }

    void
    cuda_sum_float(int nblocks, float *a, float *rtn, int nelements) {
        float *d_sum;
        int blockSize = 256;
        int numBlocks = (nblocks + blockSize * 2 - 1) / (blockSize * 2);
        vmalloc((void **) &d_sum, sizeof(float));

        cudaMemcpy(d_sum, rtn, sizeof(float), cudaMemcpyHostToDevice);
        array_sum_float<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(a, d_sum, nelements);
        cudaMemcpy(rtn, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        vfree(d_sum);
    }

    void
    cuda_prod_float(int nblocks, float *a, float *rtn, int nelements) {
        float *d_prod;
        int blockSize = 256;
        int numBlocks = (nblocks + blockSize * 2 - 1) / (blockSize * 2);
        vmalloc((void **) &d_prod, sizeof(float));

        cudaMemcpy(d_prod, rtn, sizeof(float), cudaMemcpyHostToDevice);
        array_prod_float<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(a, d_prod, nelements);
        cudaMemcpy(rtn, d_prod, sizeof(float), cudaMemcpyDeviceToHost);
        vfree(d_prod);
    }

    /* Local-only error helpers for cuda_svd_float — the surrounding
       CHECK_CUSOLVER / CHECK_CUDA macros return EXIT_FAILURE without
       releasing handles that were allocated before the failing call,
       which would leak cusolverH / stream / gesvdj_params / devInfo /
       d_work on the early-exit paths. Use FAIL_* instead so every
       failure routes through the single cleanup block at the bottom. */
#define SVD_FAIL_CUSOLVER(call) do {                                                \
        cusolverStatus_t _st = (call);                                              \
        if (_st != CUSOLVER_STATUS_SUCCESS) {                                       \
            fprintf(stderr,                                                         \
                "cuda_svd_float: cuSOLVER call failed at %s:%d (status=%d)\n",      \
                __FILE__, __LINE__, (int) _st);                                     \
            rc = -1; goto cleanup;                                                  \
        }                                                                           \
    } while (0)
#define SVD_FAIL_CUDA(call) do {                                                    \
        cudaError_t _st = (call);                                                   \
        if (_st != cudaSuccess) {                                                   \
            fprintf(stderr,                                                         \
                "cuda_svd_float: CUDA call failed at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_st));                       \
            rc = -1; goto cleanup;                                                  \
        }                                                                           \
    } while (0)
#define SVD_FAIL_VMALLOC(ptr) do {                                                  \
        if ((ptr) == NULL) {                                                        \
            fprintf(stderr,                                                         \
                "cuda_svd_float: vmalloc failed at %s:%d\n",                        \
                __FILE__, __LINE__);                                                \
            rc = -1; goto cleanup;                                                  \
        }                                                                           \
    } while (0)
    int
    cuda_svd_float(float *d_A, float *d_U, float *d_V, float *d_S, int m, int n) {
        cusolverDnHandle_t cusolverH = NULL;
        cudaStream_t stream = NULL;
        gesvdjInfo_t gesvdj_params = NULL;
        int *devInfo = NULL;
        float *d_work = NULL;
        int lwork = 0;
        int rc = 1;

        SVD_FAIL_CUSOLVER(cusolverDnCreate(&cusolverH));
        SVD_FAIL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        SVD_FAIL_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
        SVD_FAIL_CUSOLVER(cusolverDnCreateGesvdjInfo(&gesvdj_params));

        SVD_FAIL_CUSOLVER(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1.e-7));
        SVD_FAIL_CUSOLVER(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 15));

        /* Route both device scratch buffers through vmalloc so NDARRAY_VCHECK=1
           accounts for them — raw cudaMalloc was invisible to the counter. */
        vmalloc((void **) &devInfo, sizeof(int));
        SVD_FAIL_VMALLOC(devInfo);
        SVD_FAIL_CUSOLVER(cusolverDnSgesvdj_bufferSize(
                cusolverH, CUSOLVER_EIG_MODE_VECTOR, 0,
                m, n, d_A, m, d_S, d_U, m, d_V, n, &lwork, gesvdj_params));

        vmalloc((void **) &d_work, sizeof(float) * (size_t) lwork);
        SVD_FAIL_VMALLOC(d_work);
        SVD_FAIL_CUSOLVER(cusolverDnSgesvdj(
                cusolverH, CUSOLVER_EIG_MODE_VECTOR, 0,
                m, n, d_A, m, d_S, d_U, m, d_V, n,
                d_work, lwork, devInfo, gesvdj_params));

        /* cusolverDnSgesvdj runs on `stream` (cudaStreamNonBlocking), which
           is unordered with the caller's default stream. The output buffers
           d_U / d_V / d_S are read on the default stream after we return,
           so we need a barrier here. cudaStreamSynchronize is the targeted
           form (avoids syncing every other stream the user may have spun
           up); kept instead of the broader cudaDeviceSynchronize that was
           dropped elsewhere in the cuda-sync-cleanup pass. */
        SVD_FAIL_CUDA(cudaStreamSynchronize(stream));

    cleanup:
        if (d_work)        vfree(d_work);
        if (devInfo)       vfree(devInfo);
        if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
        if (cusolverH)     cusolverDnDestroy(cusolverH);
        if (stream)        cudaStreamDestroy(stream);
        return rc;
    }
#undef SVD_FAIL_VMALLOC
#undef SVD_FAIL_CUDA
#undef SVD_FAIL_CUSOLVER

    float
    cuda_max_float(float *a, int nelements) {
        int size = nelements;
        float *d_out;
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        vmalloc((void**)&d_out, sizeof(float));
        /* atomicMaxFloat merges per-thread values into *d_out — the slot
           must start at -FLT_MAX so any real element wins on the first
           CAS. Without this seed, vmalloc slab reuse hands us a stale
           value (often 0 or the last sum/prod result) and atomicMax
           never overwrites a value that's larger than the input max. */
        float neg_inf = -FLT_MAX;
        cudaMemcpy(d_out, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
        max_reduce_naive<<<blocksPerGrid, threadsPerBlock>>>(d_out, a, size);
        float max_value;
        cudaMemcpy(&max_value, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        vfree(d_out);
        return max_value;
    }

    float
    cuda_min_float(float *a, int nelements) {
        int size = nelements;
        float *d_out;
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        vmalloc((void**)&d_out, sizeof(float));
        /* See cuda_max_float — atomicMinFloat requires a +FLT_MAX seed. */
        float pos_inf = FLT_MAX;
        cudaMemcpy(d_out, &pos_inf, sizeof(float), cudaMemcpyHostToDevice);
        min_reduce_naive<<<blocksPerGrid, threadsPerBlock>>>(d_out, a, size);
        float min_value;
        cudaMemcpy(&min_value, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        vfree(d_out);
        return min_value;
    }

    int
    cuda_equal_float(int nblocks, float *a, float *b, int nelements) {
        int blockSize = 256;  /* threads per block — typical choice */
        int result = 1;
        int *d_equal = NULL;
        /* Allocate the GPU scratch slot through vmalloc so NDARRAY_VCHECK=1
           sees it; the previous raw cudaMalloc was invisible to the counter. */
        vmalloc((void **) &d_equal, sizeof(int));
        if (d_equal == NULL) {
            return 0;
        }
        cudaMemcpy(d_equal, &result, sizeof(int), cudaMemcpyHostToDevice);
        int numBlocks = (nblocks + blockSize - 1) / blockSize;
        array_equals_float<<<numBlocks, blockSize>>>(a, b, d_equal, nelements);
        cudaMemcpy(&result, d_equal, sizeof(int), cudaMemcpyDeviceToHost);
        vfree(d_equal);
        return result;
    }

    void
    cuda_pow_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        pow_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
    }

    void
    cuda_mod_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        fmodf_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
    }

    void
    cuda_multiply_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        multiply_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
    }

    void
    cuda_divide_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        divide_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
    }

    void
    cuda_subtract_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        subtract_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
    }

    void
    cuda_add_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        add_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
    }

    void
    cuda_float_sqrt(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sqrtFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_abs(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        absFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_sin(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sinFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_cos(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        cosFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_tan(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        tanFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_arcsin(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arcsinFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_arctan(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctanFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_arccos(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arccosFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_radians(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        radiansFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_degrees(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        degreesFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_sinh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sinhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_cosh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        coshFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_tanh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        tanhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_arcsinh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arcsinhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_transpose(int tiledim, int blockrows, const float *d_in, float *d_out, int width, int height) {

        dim3 grid(16, 16);
        dim3 block(16, 16);
        transposeCoalesced<<<grid, block>>>(d_in, height, width, d_out);
    }

    void
    cuda_float_arccosh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arccoshFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_arctanh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctanhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_rint(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        rintFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_fix(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        fixFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_ceil(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        ceilFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_round(int nblocks, float *d_array, float decimals) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        roundToDecimalsFloatKernel<<<numBlocks, blockSize>>>(d_array, (int)decimals, nblocks);
    }

    void
    cuda_float_floor(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        floorFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_trunc(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        truncFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_sinc(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sincFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_calculate_outer_product(int m, int n, float *a_array, float *b_array, float *r_array) {
        dim3 blockSize(16, 16);  // Number of threads per block. This is a typical choice.
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        calculateOuterProductFloat<<<gridSize, blockSize>>>(a_array, b_array, m, n, r_array);
    }

    void
    cuda_float_negate(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        negateFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_positive(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        positiveFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_reciprocal(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        reciprocalFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_sign(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        signFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
    }

    void
    cuda_float_clip(int nblocks, float *d_array, float minVal, float maxVal) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        clipFloatKernel<<<numBlocks, blockSize>>>(d_array, minVal, maxVal, nblocks);
    }

    void
    cuda_float_multiply_matrix_vector(int nblocks, float *a_array, float *b_array, float *result, int rows, int cols) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        matrixVectorMultiplyFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, rows, cols);
    }

    void
    cuda_float_compare_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
    }

    void
    cuda_float_compare_not_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysNotEqualFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
    }

    void
    cuda_float_compare_greater(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysGreaterFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
    }

    void
    cuda_float_compare_greater_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysGreaterEqualFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
    }

    float
    cuda_float_median_float(int nblocks, float *a_array, int n) {
        const int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        /* vmalloc — tracked by NDARRAY_VCHECK=1; matched vfree at exit. */
        float *d_medians = NULL;
        vmalloc((void **) &d_medians, (size_t) blocksPerGrid * sizeof(float));
        if (d_medians == NULL) {
            return 0.0f;
        }

        findMedianKernelFloat<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(a_array, n, d_medians);

        /* Perform a final reduction to find the overall median. */
        while (blocksPerGrid > 1)
        {
            int newBlocks = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
            findMedianKernelFloat<<<newBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_medians, blocksPerGrid, d_medians);
            blocksPerGrid = newBlocks;
        }

        float median;
        cudaMemcpy(&median, d_medians, sizeof(float), cudaMemcpyDeviceToHost);
        vfree(d_medians);
        return median;
    }

    void
    cuda_float_compare_less(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysLessFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
    }

    void
    cuda_float_compare_less_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysLessEqualFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
    }

    void
    cuda_convolve2d_same_float(const float* a, const float* b,
                                   const int* shape_a, const int* shape_b,
                                   const int* strides_a, const int* strides_b,
                                   char boundary, float* output,
                                   float fill_value) {
        int output_height = shape_a[0];
        int output_width = shape_a[1];
        // Configure grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x,
                     (output_height + blockDim.y - 1) / blockDim.y);

        int *d_shape_a, *d_shape_b, *d_strides_a, *d_strides_b;

        vmalloc((void**)&d_shape_a, sizeof(int) * 2);
        vmalloc((void**)&d_shape_b, sizeof(int) * 2);
        vmalloc((void**)&d_strides_a, sizeof(int) * 2);
        vmalloc((void**)&d_strides_b, sizeof(int) * 2);

        vmemcpyh2d((char*)shape_a, (char*)d_shape_a, sizeof(int) * 2);
        vmemcpyh2d((char*)shape_b, (char*)d_shape_b, sizeof(int) * 2);
        vmemcpyh2d((char*)strides_a, (char*)d_strides_a, sizeof(int) * 2);
        vmemcpyh2d((char*)strides_b, (char*)d_strides_b, sizeof(int) * 2);
        // Launch the CUDA kernel
        convolve2dSameFloatKernel<<<gridDim, blockDim>>>(a, b, d_shape_a, d_shape_b,
                                                         d_strides_a, d_strides_b, boundary,
                                                         output, fill_value);
        vfree(d_shape_a);
        vfree(d_shape_b);
        vfree(d_strides_a);
        vfree(d_strides_b);
    }

    NDArray*
    NDArrayMathGPU_ElementWise(NDArray* ndarray, ElementWiseFloatGPUOperation op) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_F32DATA(rtn));
        return rtn;
    }

    NDArray*
    NDArrayMathGPU_ElementWise1F(NDArray* ndarray, ElementWiseFloatGPUOperation1F op, float val1) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_F32DATA(rtn), val1);
        return rtn;
    }

    NDArray*
    NDArrayMathGPU_ElementWise1N(NDArray* ndarray, ElementWiseFloatGPUOperation1N op, NDArray* val1) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_F32DATA(rtn), NDArray_F32DATA(val1));
        return rtn;
    }

    NDArray*
    NDArrayMathGPU_ElementWise2F(NDArray* ndarray, ElementWiseFloatGPUOperation2F op, float val1, float val2) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_F32DATA(rtn), val1, val2);
        return rtn;
    }

    void
    cuda_float_lu(float *matrix, float *L, float *U, float *P, int size) {
        int BLOCK_SIZE = 16;
        dim3 gridSize((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

        luFloatDecompositionKernel<<<gridSize, blockSize>>>(matrix, L, U, P, size);
    }

    void
    cuda_matrix_float_l1norm(float *target, float *rtn, int rows, int cols) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

        matrixL1NormFloatKernel<<<blocksPerGrid, threadsPerBlock>>>(target, rtn, rows, cols);
    }

    /* Compute the matrix 2-norm (largest singular value) via cuSOLVER's
       SVD. Returns 0 on success, -1 on error; on error a message is
       written to stderr. The original implementation used CHECK_CUSOLVER /
       CHECK_CUDA which `return EXIT_FAILURE` without cleanup, leaking
       `handle` / `d_work` / `d_singular_values` on any partial failure.
       Rewritten with a cleanup label so every exit releases the same
       resources. */
    int
    cuda_matrix_float_l2norm(float *target, float *rtn, int rows, int cols) {
        cusolverDnHandle_t handle = NULL;
        float *d_work = NULL;
        float *d_singular_values = NULL;
        int work_size = 0;
        int rc = 0;
        cusolverStatus_t cstat;
        cudaError_t cerr;

        cstat = cusolverDnCreate(&handle);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_l2norm: cusolverDnCreate failed (status=%d)\n", (int)cstat);
            rc = -1; goto cleanup;
        }
        cstat = cusolverDnSgesvd_bufferSize(handle, rows, cols, &work_size);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_l2norm: bufferSize failed (status=%d)\n", (int)cstat);
            rc = -1; goto cleanup;
        }
        vmalloc((void **) &d_work, (size_t) work_size * sizeof(float));
        vmalloc((void **) &d_singular_values, (size_t) cols * sizeof(float));

        cstat = cusolverDnSgesvd(handle, 'N', 'N', rows, cols, target, rows,
                                 d_singular_values, NULL, rows, NULL, cols,
                                 d_work, work_size, NULL, NULL);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_l2norm: SVD failed (status=%d)\n", (int)cstat);
            rc = -1; goto cleanup;
        }
        cerr = cudaMemcpy(rtn, d_singular_values, sizeof(float), cudaMemcpyDeviceToDevice);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "cuda_matrix_float_l2norm: cudaMemcpy failed: %s\n",
                    cudaGetErrorString(cerr));
            rc = -1;
        }

    cleanup:
        if (d_work)            vfree(d_work);
        if (d_singular_values) vfree(d_singular_values);
        if (handle)            cusolverDnDestroy(handle);
        return rc;
    }

    /* Compute an in-place matrix inverse via cuSOLVER's LU decomposition
       and triangular solve. The original implementation ignored every
       cusolverDn return code and would silently produce garbage when LU
       diverged. Now all cusolver calls are checked; on any failure we
       still walk the cleanup path so VCHECK stays balanced. */
    void cuda_matrix_float_inverse(float* matrix, int n) {
        cusolverDnHandle_t cusolverH = NULL;
        int *d_info = NULL;
        int *d_pivot = NULL;
        float *d_work = NULL;
        float *d_identity = NULL;
        int lwork = 0;
        cusolverStatus_t cstat;

        cstat = cusolverDnCreate(&cusolverH);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_inverse: cusolverDnCreate failed (status=%d)\n", (int)cstat);
            goto cleanup;
        }
        vmalloc((void**) &d_info, sizeof(int));

        cstat = cusolverDnSgetrf_bufferSize(cusolverH, n, n, matrix, n, &lwork);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_inverse: getrf_bufferSize failed (status=%d)\n", (int)cstat);
            goto cleanup;
        }
        if (lwork > 0) {
            vmalloc((void**) &d_work, (size_t) lwork * sizeof(float));
        }
        vmalloc((void**) &d_pivot, (size_t) n * sizeof(int));

        cstat = cusolverDnSgetrf(cusolverH, n, n, matrix, n, d_work, d_pivot, d_info);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_inverse: LU decomposition failed (status=%d)\n", (int)cstat);
            goto cleanup;
        }

        vmalloc((void**) &d_identity, (size_t) n * (size_t) n * sizeof(float));
        cudaMemset(d_identity, 0, (size_t) n * (size_t) n * sizeof(float));
        {
            float onef = 1.0f;
            for (int i = 0; i < n; ++i) {
                cudaMemcpy(d_identity + i * n + i, &onef, sizeof(float),
                           cudaMemcpyHostToDevice);
            }
        }

        cstat = cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, matrix, n,
                                 d_pivot, d_identity, n, d_info);
        if (cstat != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cuda_matrix_float_inverse: triangular solve failed (status=%d)\n", (int)cstat);
            goto cleanup;
        }
        cudaMemcpy(matrix, d_identity, (size_t) n * (size_t) n * sizeof(float),
                   cudaMemcpyDeviceToHost);

    cleanup:
        if (d_identity) vfree(d_identity);
        if (d_pivot)    vfree(d_pivot);
        if (d_work)     vfree(d_work);
        if (d_info)     vfree(d_info);
        if (cusolverH)  cusolverDnDestroy(cusolverH);
    }

    void cuda_matrix_eig_float(float* d_matrix, int n, float* d_eigvalues) {
        cusolverDnHandle_t handle = NULL;
        cusolverStatus_t status;
        int* d_info = NULL;
        float* d_work = NULL;

        /* Allocate d_info via vmalloc so VCHECK tracks it; cusolverDnCreate
           may fail before any work is queued, but the d_info slot must still
           be released on every exit path. */
        vmalloc((void**)&d_info, sizeof(int));

        status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            printf("CUSOLVER initialization failed.\n");
            vfree(d_info);
            return;
        }

        int lwork = 0;
        status = cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_UPPER, n, d_matrix,
                                             n, d_eigvalues, &lwork);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            printf("CUSOLVER workspace size computation failed.\n");
            cusolverDnDestroy(handle);
            vfree(d_info);
            return;
        }

        vmalloc((void**)&d_work, lwork * sizeof(float));

        status = cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_UPPER, n, d_matrix, n,
                                  d_eigvalues, d_work, lwork, d_info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            printf("CUSOLVER eigenvectors computation failed.\n");
            vfree(d_work);
            cusolverDnDestroy(handle);
            vfree(d_info);
            return;
        }

        vfree(d_work);
        cusolverDnDestroy(handle);
        vfree(d_info);
    }

    void cuda_sum_double(int nblocks, double *a, double *rtn, int nelements) {
        double *d_sum;
        int blockSize = 256;
        int numBlocks = (nblocks + blockSize * 2 - 1) / (blockSize * 2);
        vmalloc((void **) &d_sum, sizeof(double));

        cudaMemcpy(d_sum, rtn, sizeof(double), cudaMemcpyHostToDevice);
        array_sum_reduce_blocks<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(a, d_sum, nelements);
        finalize_sum<<<1, 1>>>(d_sum, rtn, numBlocks);
        cudaMemcpy(rtn, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
        vfree(d_sum);
    }

/* ──────────────────────────────────────────────────────────────────────────
   Typed wrapper functions exposed to the C side.

   Naming: cuda_<op>_<dtype>(a, b, rtn, n)
   For dtype ∈ {i8, u8, i16, u16, i32, u32, i64, u64, f16, f64, dd128}.
   The dd128 wrappers operate on the GPU's double-double layout (2*n doubles
   contiguous: hi[0], lo[0], hi[1], lo[1], ...).
   ────────────────────────────────────────────────────────────────────────── */

#define GRID_FOR(n)  int blockSize = 256; int numBlocks = ((n) + blockSize - 1) / blockSize

/* Wrappers stay async on the default stream — the next cudaMemcpy or kernel
   launched in the same stream implicitly orders behind this kernel. The
   user-facing NumPower::syncDevice() is the explicit barrier. */
#define DEF_BINOP_WRAPPER(NAME, KERNEL, T)                                          \
void NAME(T *a, T *b, T *rtn, int n) {                                              \
    GRID_FOR(n);                                                                    \
    KERNEL<T><<<numBlocks, blockSize>>>(a, b, rtn, n);                              \
}

DEF_BINOP_WRAPPER(cuda_add_i8,  tcuda_add_kernel, int8_t)
DEF_BINOP_WRAPPER(cuda_sub_i8,  tcuda_sub_kernel, int8_t)
DEF_BINOP_WRAPPER(cuda_mul_i8,  tcuda_mul_kernel, int8_t)
DEF_BINOP_WRAPPER(cuda_div_i8,  tcuda_div_kernel, int8_t)
DEF_BINOP_WRAPPER(cuda_add_u8,  tcuda_add_kernel, uint8_t)
DEF_BINOP_WRAPPER(cuda_sub_u8,  tcuda_sub_kernel, uint8_t)
DEF_BINOP_WRAPPER(cuda_mul_u8,  tcuda_mul_kernel, uint8_t)
DEF_BINOP_WRAPPER(cuda_div_u8,  tcuda_div_kernel, uint8_t)
DEF_BINOP_WRAPPER(cuda_add_i16, tcuda_add_kernel, int16_t)
DEF_BINOP_WRAPPER(cuda_sub_i16, tcuda_sub_kernel, int16_t)
DEF_BINOP_WRAPPER(cuda_mul_i16, tcuda_mul_kernel, int16_t)
DEF_BINOP_WRAPPER(cuda_div_i16, tcuda_div_kernel, int16_t)
DEF_BINOP_WRAPPER(cuda_add_u16, tcuda_add_kernel, uint16_t)
DEF_BINOP_WRAPPER(cuda_sub_u16, tcuda_sub_kernel, uint16_t)
DEF_BINOP_WRAPPER(cuda_mul_u16, tcuda_mul_kernel, uint16_t)
DEF_BINOP_WRAPPER(cuda_div_u16, tcuda_div_kernel, uint16_t)
DEF_BINOP_WRAPPER(cuda_add_i32, tcuda_add_kernel, int32_t)
DEF_BINOP_WRAPPER(cuda_sub_i32, tcuda_sub_kernel, int32_t)
DEF_BINOP_WRAPPER(cuda_mul_i32, tcuda_mul_kernel, int32_t)
DEF_BINOP_WRAPPER(cuda_div_i32, tcuda_div_kernel, int32_t)
DEF_BINOP_WRAPPER(cuda_add_u32, tcuda_add_kernel, uint32_t)
DEF_BINOP_WRAPPER(cuda_sub_u32, tcuda_sub_kernel, uint32_t)
DEF_BINOP_WRAPPER(cuda_mul_u32, tcuda_mul_kernel, uint32_t)
DEF_BINOP_WRAPPER(cuda_div_u32, tcuda_div_kernel, uint32_t)
DEF_BINOP_WRAPPER(cuda_add_i64, tcuda_add_kernel, int64_t)
DEF_BINOP_WRAPPER(cuda_sub_i64, tcuda_sub_kernel, int64_t)
DEF_BINOP_WRAPPER(cuda_mul_i64, tcuda_mul_kernel, int64_t)
DEF_BINOP_WRAPPER(cuda_div_i64, tcuda_div_kernel, int64_t)
DEF_BINOP_WRAPPER(cuda_add_u64, tcuda_add_kernel, uint64_t)
DEF_BINOP_WRAPPER(cuda_sub_u64, tcuda_sub_kernel, uint64_t)
DEF_BINOP_WRAPPER(cuda_mul_u64, tcuda_mul_kernel, uint64_t)
DEF_BINOP_WRAPPER(cuda_div_u64, tcuda_div_kernel, uint64_t)
DEF_BINOP_WRAPPER(cuda_add_f64, tcuda_add_kernel, double)
DEF_BINOP_WRAPPER(cuda_sub_f64, tcuda_sub_kernel, double)
DEF_BINOP_WRAPPER(cuda_mul_f64, tcuda_mul_kernel, double)
DEF_BINOP_WRAPPER(cuda_div_f64, tcuda_div_kernel, double)

#define DEF_MOD_INT_WRAPPER(NAME, T)                                                \
void NAME(T *a, T *b, T *rtn, int n) {                                              \
    GRID_FOR(n);                                                                    \
    tcuda_mod_int_kernel<T><<<numBlocks, blockSize>>>(a, b, rtn, n);                \
}
DEF_MOD_INT_WRAPPER(cuda_mod_i8,  int8_t)
DEF_MOD_INT_WRAPPER(cuda_mod_u8,  uint8_t)
DEF_MOD_INT_WRAPPER(cuda_mod_i16, int16_t)
DEF_MOD_INT_WRAPPER(cuda_mod_u16, uint16_t)
DEF_MOD_INT_WRAPPER(cuda_mod_i32, int32_t)
DEF_MOD_INT_WRAPPER(cuda_mod_u32, uint32_t)
DEF_MOD_INT_WRAPPER(cuda_mod_i64, int64_t)
DEF_MOD_INT_WRAPPER(cuda_mod_u64, uint64_t)
void cuda_mod_f64(double *a, double *b, double *rtn, int n) {
    GRID_FOR(n);
    tcuda_mod_f64_kernel<<<numBlocks, blockSize>>>(a, b, rtn, n);
}

#define DEF_POW_SIGNED_WRAPPER(NAME, T)                                             \
void NAME(T *a, T *b, T *rtn, int n) {                                              \
    GRID_FOR(n);                                                                    \
    tcuda_pow_signed_kernel<T><<<numBlocks, blockSize>>>(a, b, rtn, n);             \
}
#define DEF_POW_UNSIGNED_WRAPPER(NAME, T)                                           \
void NAME(T *a, T *b, T *rtn, int n) {                                              \
    GRID_FOR(n);                                                                    \
    tcuda_pow_unsigned_kernel<T><<<numBlocks, blockSize>>>(a, b, rtn, n);           \
}
DEF_POW_SIGNED_WRAPPER(cuda_pow_i8,  int8_t)
DEF_POW_SIGNED_WRAPPER(cuda_pow_i16, int16_t)
DEF_POW_SIGNED_WRAPPER(cuda_pow_i32, int32_t)
DEF_POW_SIGNED_WRAPPER(cuda_pow_i64, int64_t)
DEF_POW_UNSIGNED_WRAPPER(cuda_pow_u8,  uint8_t)
DEF_POW_UNSIGNED_WRAPPER(cuda_pow_u16, uint16_t)
DEF_POW_UNSIGNED_WRAPPER(cuda_pow_u32, uint32_t)
DEF_POW_UNSIGNED_WRAPPER(cuda_pow_u64, uint64_t)
void cuda_pow_f64(double *a, double *b, double *rtn, int n) {
    GRID_FOR(n);
    tcuda_pow_f64_kernel<<<numBlocks, blockSize>>>(a, b, rtn, n);
}
void cuda_atan2_f32(float *a, float *b, float *rtn, int n) {
    GRID_FOR(n);
    tcuda_atan2_kernel<float><<<numBlocks, blockSize>>>(a, b, rtn, n);
}
void cuda_atan2_f64(double *a, double *b, double *rtn, int n) {
    GRID_FOR(n);
    tcuda_atan2_kernel<double><<<numBlocks, blockSize>>>(a, b, rtn, n);
}

/* float16 (__half) wrappers */
void cuda_add_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n) {
    GRID_FOR(n);
    tcuda_add_half_kernel<<<numBlocks, blockSize>>>((__half *)a, (__half *)b, (__half *)rtn, n);
}
void cuda_sub_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n) {
    GRID_FOR(n);
    tcuda_sub_half_kernel<<<numBlocks, blockSize>>>((__half *)a, (__half *)b, (__half *)rtn, n);
}
void cuda_mul_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n) {
    GRID_FOR(n);
    tcuda_mul_half_kernel<<<numBlocks, blockSize>>>((__half *)a, (__half *)b, (__half *)rtn, n);
}
void cuda_div_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n) {
    GRID_FOR(n);
    tcuda_div_half_kernel<<<numBlocks, blockSize>>>((__half *)a, (__half *)b, (__half *)rtn, n);
}
void cuda_mod_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n) {
    GRID_FOR(n);
    tcuda_mod_half_kernel<<<numBlocks, blockSize>>>((__half *)a, (__half *)b, (__half *)rtn, n);
}
void cuda_pow_f16(uint16_t *a, uint16_t *b, uint16_t *rtn, int n) {
    GRID_FOR(n);
    tcuda_pow_half_kernel<<<numBlocks, blockSize>>>((__half *)a, (__half *)b, (__half *)rtn, n);
}

/* Typed fills */
#define DEF_FILL_WRAPPER(NAME, T)                                                   \
void NAME(T *a, T value, int n) {                                                   \
    GRID_FOR(n);                                                                    \
    tcuda_fill_kernel<T><<<numBlocks, blockSize>>>(a, value, n);                    \
}
DEF_FILL_WRAPPER(cuda_fill_i8,  int8_t)
DEF_FILL_WRAPPER(cuda_fill_u8,  uint8_t)
DEF_FILL_WRAPPER(cuda_fill_i16, int16_t)
DEF_FILL_WRAPPER(cuda_fill_u16, uint16_t)
DEF_FILL_WRAPPER(cuda_fill_i32, int32_t)
DEF_FILL_WRAPPER(cuda_fill_u32, uint32_t)
DEF_FILL_WRAPPER(cuda_fill_i64, int64_t)
DEF_FILL_WRAPPER(cuda_fill_u64, uint64_t)
DEF_FILL_WRAPPER(cuda_fill_f64, double)
void cuda_fill_f16(uint16_t *a, float value, int n) {
    GRID_FOR(n);
    tcuda_fill_half_kernel<<<numBlocks, blockSize>>>((__half *)a, value, n);
}
void cuda_fill_dd(double *out, double hi, double lo, int n) {
    GRID_FOR(n);
    tcuda_fill_dd_kernel<<<numBlocks, blockSize>>>(out, hi, lo, n);
}

/* GPU AsType — explicit Src→Dst cast kernels covering every pair we may need.
   The wrapper takes both src and dst typed pointers; the source is read-only.
   To keep the API surface manageable we provide one cast function that takes
   src and dst dtype tags as strings and dispatches internally; the dispatcher
   lives in cuda_math.cu via a per-pair switch. */

#define CAST_PAIR(SRC, DST, ST, DT)                                                 \
void cuda_cast_##SRC##_to_##DST(ST *src, DT *dst, int n) {                          \
    GRID_FOR(n);                                                                    \
    tcuda_cast_kernel<ST, DT><<<numBlocks, blockSize>>>(src, dst, n);               \
}

#define DECL_CAST_TARGETS(SRC, ST)                                                  \
CAST_PAIR(SRC, i8,  ST, int8_t)                                                     \
CAST_PAIR(SRC, u8,  ST, uint8_t)                                                    \
CAST_PAIR(SRC, i16, ST, int16_t)                                                    \
CAST_PAIR(SRC, u16, ST, uint16_t)                                                   \
CAST_PAIR(SRC, i32, ST, int32_t)                                                    \
CAST_PAIR(SRC, u32, ST, uint32_t)                                                   \
CAST_PAIR(SRC, i64, ST, int64_t)                                                    \
CAST_PAIR(SRC, u64, ST, uint64_t)                                                   \
CAST_PAIR(SRC, f32, ST, float)                                                      \
CAST_PAIR(SRC, f64, ST, double)

DECL_CAST_TARGETS(i8,  int8_t)
DECL_CAST_TARGETS(u8,  uint8_t)
DECL_CAST_TARGETS(i16, int16_t)
DECL_CAST_TARGETS(u16, uint16_t)
DECL_CAST_TARGETS(i32, int32_t)
DECL_CAST_TARGETS(u32, uint32_t)
DECL_CAST_TARGETS(i64, int64_t)
DECL_CAST_TARGETS(u64, uint64_t)
DECL_CAST_TARGETS(f32, float)
DECL_CAST_TARGETS(f64, double)

/* Casts involving __half — must go through __half2float / __float2half. */
#define CAST_FROM_HALF(DST, DT)                                                     \
void cuda_cast_f16_to_##DST(uint16_t *src, DT *dst, int n) {                        \
    GRID_FOR(n);                                                                    \
    tcuda_cast_from_half_kernel<DT><<<numBlocks, blockSize>>>((__half *)src, dst, n); \
}
CAST_FROM_HALF(i8,  int8_t)
CAST_FROM_HALF(u8,  uint8_t)
CAST_FROM_HALF(i16, int16_t)
CAST_FROM_HALF(u16, uint16_t)
CAST_FROM_HALF(i32, int32_t)
CAST_FROM_HALF(u32, uint32_t)
CAST_FROM_HALF(i64, int64_t)
CAST_FROM_HALF(u64, uint64_t)
CAST_FROM_HALF(f32, float)
CAST_FROM_HALF(f64, double)

#define CAST_TO_HALF(SRC, ST)                                                       \
void cuda_cast_##SRC##_to_f16(ST *src, uint16_t *dst, int n) {                      \
    GRID_FOR(n);                                                                    \
    tcuda_cast_to_half_kernel<ST><<<numBlocks, blockSize>>>(src, (__half *)dst, n); \
}
CAST_TO_HALF(i8,  int8_t)
CAST_TO_HALF(u8,  uint8_t)
CAST_TO_HALF(i16, int16_t)
CAST_TO_HALF(u16, uint16_t)
CAST_TO_HALF(i32, int32_t)
CAST_TO_HALF(u32, uint32_t)
CAST_TO_HALF(i64, int64_t)
CAST_TO_HALF(u64, uint64_t)
CAST_TO_HALF(f32, float)
CAST_TO_HALF(f64, double)
void cuda_cast_f16_to_f16(uint16_t *src, uint16_t *dst, int n) {
    GRID_FOR(n);
    tcuda_cast_half_to_half_kernel<<<numBlocks, blockSize>>>((__half *)src, (__half *)dst, n);
}

/* dd128 wrappers — note the buffer holds 2n doubles (hi, lo interleaved). */
#define DEF_DD_WRAPPER(NAME, KERNEL)                                                \
void NAME(double *a, double *b, double *rtn, int n) {                               \
    GRID_FOR(n);                                                                    \
    KERNEL<<<numBlocks, blockSize>>>(a, b, rtn, n);                                 \
}
DEF_DD_WRAPPER(cuda_add_dd, tcuda_add_dd_kernel)
DEF_DD_WRAPPER(cuda_sub_dd, tcuda_sub_dd_kernel)
DEF_DD_WRAPPER(cuda_mul_dd, tcuda_mul_dd_kernel)
DEF_DD_WRAPPER(cuda_div_dd, tcuda_div_dd_kernel)
DEF_DD_WRAPPER(cuda_pow_dd, tcuda_pow_dd_kernel)
DEF_DD_WRAPPER(cuda_mod_dd, tcuda_mod_dd_kernel)
DEF_DD_WRAPPER(cuda_atan2_dd, tcuda_atan2_dd_kernel)

/* ──────────────────────────────────────────────────────────────────────────
   Typed unary op wrappers — element-wise abs / negate / sign / square /
   clip / sqrt / rsqrt / reciprocal / sinc. Naming convention:
     cuda_<op>_<dtype>(buffer, [extra args], n)
   The buffer is updated in place; the dispatcher (`NDArray_TypedUnaryOp`)
   allocates the output via `NDArray_Copy` first when the op preserves
   dtype, or via `NDArray_AsType` when the op promotes integer → float.
   ────────────────────────────────────────────────────────────────────────── */

#define DEF_UNOP_T_WRAPPER(NAME, KERNEL, T)                                         \
void NAME(T *a, int n) {                                                            \
    GRID_FOR(n);                                                                    \
    KERNEL<T><<<numBlocks, blockSize>>>(a, n);                                      \
}

DEF_UNOP_T_WRAPPER(cuda_negate_i8,  tcuda_negate_kernel, int8_t)
DEF_UNOP_T_WRAPPER(cuda_negate_u8,  tcuda_negate_kernel, uint8_t)
DEF_UNOP_T_WRAPPER(cuda_negate_i16, tcuda_negate_kernel, int16_t)
DEF_UNOP_T_WRAPPER(cuda_negate_u16, tcuda_negate_kernel, uint16_t)
DEF_UNOP_T_WRAPPER(cuda_negate_i32, tcuda_negate_kernel, int32_t)
DEF_UNOP_T_WRAPPER(cuda_negate_u32, tcuda_negate_kernel, uint32_t)
DEF_UNOP_T_WRAPPER(cuda_negate_i64, tcuda_negate_kernel, int64_t)
DEF_UNOP_T_WRAPPER(cuda_negate_u64, tcuda_negate_kernel, uint64_t)
DEF_UNOP_T_WRAPPER(cuda_negate_f64, tcuda_negate_kernel, double)
DEF_UNOP_T_WRAPPER(cuda_negate_f32, tcuda_negate_kernel, float)
void cuda_negate_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_negate_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_negate_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_negate_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

/* Signed-int `abs` uses the wrapping kernel; unsigned `abs` is a no-op so
   the dispatcher omits it. */
DEF_UNOP_T_WRAPPER(cuda_abs_i8,  tcuda_abs_int_kernel, int8_t)
DEF_UNOP_T_WRAPPER(cuda_abs_i16, tcuda_abs_int_kernel, int16_t)
DEF_UNOP_T_WRAPPER(cuda_abs_i32, tcuda_abs_int_kernel, int32_t)
DEF_UNOP_T_WRAPPER(cuda_abs_i64, tcuda_abs_int_kernel, int64_t)
DEF_UNOP_T_WRAPPER(cuda_abs_f32, tcuda_abs_float_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_abs_f64, tcuda_abs_float_kernel, double)
void cuda_abs_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_abs_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_abs_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_abs_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

DEF_UNOP_T_WRAPPER(cuda_sign_i8,  tcuda_sign_kernel, int8_t)
DEF_UNOP_T_WRAPPER(cuda_sign_u8,  tcuda_sign_kernel, uint8_t)
DEF_UNOP_T_WRAPPER(cuda_sign_i16, tcuda_sign_kernel, int16_t)
DEF_UNOP_T_WRAPPER(cuda_sign_u16, tcuda_sign_kernel, uint16_t)
DEF_UNOP_T_WRAPPER(cuda_sign_i32, tcuda_sign_kernel, int32_t)
DEF_UNOP_T_WRAPPER(cuda_sign_u32, tcuda_sign_kernel, uint32_t)
DEF_UNOP_T_WRAPPER(cuda_sign_i64, tcuda_sign_kernel, int64_t)
DEF_UNOP_T_WRAPPER(cuda_sign_u64, tcuda_sign_kernel, uint64_t)
DEF_UNOP_T_WRAPPER(cuda_sign_f32, tcuda_sign_float_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_sign_f64, tcuda_sign_float_kernel, double)
void cuda_sign_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_sign_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_sign_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_sign_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

DEF_UNOP_T_WRAPPER(cuda_square_i8,  tcuda_square_kernel, int8_t)
DEF_UNOP_T_WRAPPER(cuda_square_u8,  tcuda_square_kernel, uint8_t)
DEF_UNOP_T_WRAPPER(cuda_square_i16, tcuda_square_kernel, int16_t)
DEF_UNOP_T_WRAPPER(cuda_square_u16, tcuda_square_kernel, uint16_t)
DEF_UNOP_T_WRAPPER(cuda_square_i32, tcuda_square_kernel, int32_t)
DEF_UNOP_T_WRAPPER(cuda_square_u32, tcuda_square_kernel, uint32_t)
DEF_UNOP_T_WRAPPER(cuda_square_i64, tcuda_square_kernel, int64_t)
DEF_UNOP_T_WRAPPER(cuda_square_u64, tcuda_square_kernel, uint64_t)
/* Float square goes through the same kernel — multiplication is well-defined
   on every float dtype and matches `x * x` exactly. */
DEF_UNOP_T_WRAPPER(cuda_square_f32, tcuda_square_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_square_f64, tcuda_square_kernel, double)
void cuda_square_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_square_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_square_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_square_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

DEF_UNOP_T_WRAPPER(cuda_recip_f32, tcuda_recip_float_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_recip_f64, tcuda_recip_float_kernel, double)
void cuda_recip_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_recip_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_recip_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_recip_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

DEF_UNOP_T_WRAPPER(cuda_sqrt_f32, tcuda_sqrt_float_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_sqrt_f64, tcuda_sqrt_float_kernel, double)
void cuda_sqrt_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_sqrt_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_sqrt_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_sqrt_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

DEF_UNOP_T_WRAPPER(cuda_rsqrt_f32, tcuda_rsqrt_float_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_rsqrt_f64, tcuda_rsqrt_float_kernel, double)
void cuda_rsqrt_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_rsqrt_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_rsqrt_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_rsqrt_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

DEF_UNOP_T_WRAPPER(cuda_sinc_f32, tcuda_sinc_float_kernel, float)
DEF_UNOP_T_WRAPPER(cuda_sinc_f64, tcuda_sinc_float_kernel, double)
void cuda_sinc_f16(uint16_t *a, int n) {
    GRID_FOR(n);
    tcuda_sinc_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);
}
void cuda_sinc_dd(double *a, int n) {
    GRID_FOR(n);
    tcuda_sinc_dd_kernel<<<numBlocks, blockSize>>>(a, n);
}

/* ── Transcendental wrappers (exp / exp2 / expm1 / log / log1p / log2 /
   log10 / logb). Each op exposes a `_f32`, `_f64`, `_f16`, and `_dd`
   wrapper so the host dispatcher can route by compute dtype without
   re-templating. */
#define DEF_TRANSC_WRAPPERS(OP, KERNEL)                                              \
DEF_UNOP_T_WRAPPER(cuda_##OP##_f32, KERNEL##_float_kernel, float)                    \
DEF_UNOP_T_WRAPPER(cuda_##OP##_f64, KERNEL##_float_kernel, double)                   \
void cuda_##OP##_f16(uint16_t *a, int n) {                                           \
    GRID_FOR(n);                                                                     \
    KERNEL##_half_kernel<<<numBlocks, blockSize>>>((__half *)a, n);                  \
}                                                                                    \
void cuda_##OP##_dd(double *a, int n) {                                              \
    GRID_FOR(n);                                                                     \
    KERNEL##_dd_kernel<<<numBlocks, blockSize>>>(a, n);                              \
}

DEF_TRANSC_WRAPPERS(exp,   tcuda_exp)
DEF_TRANSC_WRAPPERS(exp2,  tcuda_exp2)
DEF_TRANSC_WRAPPERS(expm1, tcuda_expm1)
DEF_TRANSC_WRAPPERS(log,   tcuda_log)
DEF_TRANSC_WRAPPERS(log1p, tcuda_log1p)
DEF_TRANSC_WRAPPERS(log2,  tcuda_log2)
DEF_TRANSC_WRAPPERS(log10, tcuda_log10)
DEF_TRANSC_WRAPPERS(logb,  tcuda_logb)

/* Trig / hyperbolic / angle / rounding wrappers — same naming and
   shape as the transcendental wrappers above so the host dispatcher
   uses a single template (cf. `UNOP_GPU_TRIG_DT` in arithmetics.c). */
DEF_TRANSC_WRAPPERS(sin,      tcuda_sin)
DEF_TRANSC_WRAPPERS(cos,      tcuda_cos)
DEF_TRANSC_WRAPPERS(tan,      tcuda_tan)
DEF_TRANSC_WRAPPERS(arcsin,   tcuda_arcsin)
DEF_TRANSC_WRAPPERS(arccos,   tcuda_arccos)
DEF_TRANSC_WRAPPERS(arctan,   tcuda_arctan)
DEF_TRANSC_WRAPPERS(sinh,     tcuda_sinh)
DEF_TRANSC_WRAPPERS(cosh,     tcuda_cosh)
DEF_TRANSC_WRAPPERS(tanh,     tcuda_tanh)
DEF_TRANSC_WRAPPERS(arcsinh,  tcuda_arcsinh)
DEF_TRANSC_WRAPPERS(arccosh,  tcuda_arccosh)
DEF_TRANSC_WRAPPERS(arctanh,  tcuda_arctanh)
DEF_TRANSC_WRAPPERS(degrees,  tcuda_degrees)
DEF_TRANSC_WRAPPERS(radians,  tcuda_radians)
DEF_TRANSC_WRAPPERS(rint,     tcuda_rint)
DEF_TRANSC_WRAPPERS(trunc,    tcuda_trunc)
DEF_TRANSC_WRAPPERS(floor,    tcuda_floor)
DEF_TRANSC_WRAPPERS(ceil,     tcuda_ceil)

#undef DEF_TRANSC_WRAPPERS

#define DEF_CLIP_T_WRAPPER(NAME, T)                                                 \
void NAME(T *a, T lo, T hi, int n) {                                                \
    GRID_FOR(n);                                                                    \
    tcuda_clip_kernel<T><<<numBlocks, blockSize>>>(a, lo, hi, n);                   \
}
DEF_CLIP_T_WRAPPER(cuda_clip_i8,  int8_t)
DEF_CLIP_T_WRAPPER(cuda_clip_u8,  uint8_t)
DEF_CLIP_T_WRAPPER(cuda_clip_i16, int16_t)
DEF_CLIP_T_WRAPPER(cuda_clip_u16, uint16_t)
DEF_CLIP_T_WRAPPER(cuda_clip_i32, int32_t)
DEF_CLIP_T_WRAPPER(cuda_clip_u32, uint32_t)
DEF_CLIP_T_WRAPPER(cuda_clip_i64, int64_t)
DEF_CLIP_T_WRAPPER(cuda_clip_u64, uint64_t)
DEF_CLIP_T_WRAPPER(cuda_clip_f32, float)
DEF_CLIP_T_WRAPPER(cuda_clip_f64, double)
void cuda_clip_f16(uint16_t *a, float lo, float hi, int n) {
    GRID_FOR(n);
    tcuda_clip_half_kernel<<<numBlocks, blockSize>>>((__half *)a, lo, hi, n);
}
void cuda_clip_dd(double *a, double lo_hi, double lo_lo,
                  double hi_hi, double hi_lo, int n) {
    GRID_FOR(n);
    tcuda_clip_dd_kernel<<<numBlocks, blockSize>>>(a, lo_hi, lo_lo, hi_hi, hi_lo, n);
}

/* float4 / float8 GPU casts. Names mirror the cuda_cast_<src>_to_<dst>
   convention. fp4 / fp8 storage on GPU is one byte per element. */
#define DEF_FPSMALL_CAST(NAME, KERNEL, ST, DT)                                      \
void NAME(ST *src, DT *dst, int n) {                                                \
    GRID_FOR(n);                                                                    \
    KERNEL<<<numBlocks, blockSize>>>(src, dst, n);                                  \
}

void cuda_cast_fp4_to_f32(uint8_t *src, float *dst, int n) {
    GRID_FOR(n); tcuda_cast_fp4_to_f32_kernel<<<numBlocks, blockSize>>>(src, dst, n);
}
void cuda_cast_f32_to_fp4(float *src, uint8_t *dst, int n) {
    GRID_FOR(n); tcuda_cast_f32_to_fp4_kernel<<<numBlocks, blockSize>>>(src, dst, n);
}
void cuda_cast_fp4_to_f16(uint8_t *src, uint16_t *dst, int n) {
    GRID_FOR(n);
    tcuda_cast_fp4_to_f16_kernel<<<numBlocks, blockSize>>>(src, (__half *)dst, n);
}
void cuda_cast_f16_to_fp4(uint16_t *src, uint8_t *dst, int n) {
    GRID_FOR(n);
    tcuda_cast_f16_to_fp4_kernel<<<numBlocks, blockSize>>>((__half *)src, dst, n);
}
void cuda_cast_fp8_to_f32(uint8_t *src, float *dst, int n) {
    GRID_FOR(n); tcuda_cast_fp8_to_f32_kernel<<<numBlocks, blockSize>>>(src, dst, n);
}
void cuda_cast_f32_to_fp8(float *src, uint8_t *dst, int n) {
    GRID_FOR(n); tcuda_cast_f32_to_fp8_kernel<<<numBlocks, blockSize>>>(src, dst, n);
}
void cuda_cast_fp8_to_f16(uint8_t *src, uint16_t *dst, int n) {
    GRID_FOR(n);
    tcuda_cast_fp8_to_f16_kernel<<<numBlocks, blockSize>>>(src, (__half *)dst, n);
}
void cuda_cast_f16_to_fp8(uint16_t *src, uint8_t *dst, int n) {
    GRID_FOR(n);
    tcuda_cast_f16_to_fp8_kernel<<<numBlocks, blockSize>>>((__half *)src, dst, n);
}

/* Broadcast wrapper. Caller must have:
   - src GPU buffer of elsize * src_n bytes
   - dst GPU buffer of elsize * n_out bytes (pre-allocated, untouched)
   - src_offsets_gpu: GPU buffer of n_out ints, each entry < src_n. Computed
     host-side from the broadcast rule and copied via cudaMemcpy beforehand. */
void cuda_broadcast(const char *src_gpu, char *dst_gpu,
                    const int *src_offsets_gpu, int n_out, int elsize) {
    int blockSize = 256;
    int numBlocks = (n_out + blockSize - 1) / blockSize;
    tcuda_broadcast_by_offsets<<<numBlocks, blockSize>>>(
        src_gpu, dst_gpu, src_offsets_gpu, n_out, elsize);
}

/* ── Typed reduction wrappers ──────────────────────────────────────────────
   The C-side reducer (NDArray_Reduce_*) calls one of these per dtype with
   a GPU-resident double slot pre-seeded with the operation's identity.
   The kernels do per-block tree reduction on shared memory (no warp
   shuffles, no UB across compute capabilities), then thread 0 of each
   block atomically merges the partial into the global slot. Grid is
   capped at REDUCE_MAX_BLOCKS so atomic contention stays bounded even
   for n in the billions; the grid-stride loop inside each thread covers
   the remaining elements. Identity seeding is the caller's responsibility
   (sum: 0, prod: 1, min: +DBL_MAX, max: -DBL_MAX). */
#define REDUCE_GRID(n_)                                                             \
    int blockSize = REDUCE_BLOCK;                                                   \
    int numBlocks = ((n_) + blockSize - 1) / blockSize;                             \
    if (numBlocks > REDUCE_MAX_BLOCKS) numBlocks = REDUCE_MAX_BLOCKS

#define DEF_REDUCE_WRAPPER_TYPED(NAME, KERNEL, T)                                   \
void NAME(const T *a, double *result, int n) {                                      \
    REDUCE_GRID(n);                                                                 \
    KERNEL<T><<<numBlocks, blockSize>>>(a, result, n);                              \
}
#define DEF_REDUCE_WRAPPER_T_BUNDLE(OP, KERNEL)                                     \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_i8,  KERNEL, int8_t)                \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_u8,  KERNEL, uint8_t)               \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_i16, KERNEL, int16_t)               \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_u16, KERNEL, uint16_t)              \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_i32, KERNEL, int32_t)               \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_u32, KERNEL, uint32_t)              \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_i64, KERNEL, int64_t)               \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_u64, KERNEL, uint64_t)              \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_f32, KERNEL, float)                 \
    DEF_REDUCE_WRAPPER_TYPED(cuda_reduce_##OP##_f64, KERNEL, double)

DEF_REDUCE_WRAPPER_T_BUNDLE(sum,  tcuda_reduce_sum_kernel)
DEF_REDUCE_WRAPPER_T_BUNDLE(prod, tcuda_reduce_prod_kernel)
DEF_REDUCE_WRAPPER_T_BUNDLE(min,  tcuda_reduce_min_kernel)
DEF_REDUCE_WRAPPER_T_BUNDLE(max,  tcuda_reduce_max_kernel)

/* Emulated-dtype wrappers. The C-facing prototypes in cuda_math.h take
   the raw storage type (uint16_t for float16, uint8_t for fp4/fp8,
   double for dd128 — interleaved hi/lo) so callers can pass
   NDArray_DATA() without dtype-specific casts; we cast to the kernel's
   native pointer type inside. */
void cuda_reduce_sum_f16(const uint16_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_sum_half_kernel<<<numBlocks, blockSize>>>((const __half *)a, result, n);
}
void cuda_reduce_prod_f16(const uint16_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_prod_half_kernel<<<numBlocks, blockSize>>>((const __half *)a, result, n);
}
void cuda_reduce_min_f16(const uint16_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_min_half_kernel<<<numBlocks, blockSize>>>((const __half *)a, result, n);
}
void cuda_reduce_max_f16(const uint16_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_max_half_kernel<<<numBlocks, blockSize>>>((const __half *)a, result, n);
}
void cuda_reduce_sum_fp4(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_sum_fp4_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_prod_fp4(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_prod_fp4_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_min_fp4(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_min_fp4_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_max_fp4(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_max_fp4_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_sum_fp8(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_sum_fp8_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_prod_fp8(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_prod_fp8_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_min_fp8(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_min_fp8_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_max_fp8(const uint8_t *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_max_fp8_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_sum_dd(const double *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_sum_dd_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_prod_dd(const double *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_prod_dd_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_min_dd(const double *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_min_dd_kernel<<<numBlocks, blockSize>>>(a, result, n);
}
void cuda_reduce_max_dd(const double *a, double *result, int n) {
    REDUCE_GRID(n); tcuda_reduce_max_dd_kernel<<<numBlocks, blockSize>>>(a, result, n);
}

/* Strided device-to-device copy wrapper. Single kernel launch covers any
   slice pattern (any ndim ≤ 16, any signed strides). Returns 0 on success,
   -1 if the call falls outside the kernel's supported envelope so the
   caller can fall back to a per-row cudaMemcpy loop.
   - dst_gpu, src_gpu: device pointers; dst has space for n_elems * elsize.
   - host_shape, host_strides_b: per-axis shape and stride (bytes), read from
     host memory and packed into the kernel parameter struct.
   - n_elems: total number of output elements (product of host_shape).
   - ndim: number of axes; must be in [1, STRIDED_COPY_MAX_NDIM].
   - elsize: dtype byte size; fast paths cover 1/2/4/8/16, others use a byte loop. */
int cuda_strided_copy(char *dst_gpu, const char *src_gpu,
                      long long n_elems, int ndim, int elsize,
                      const int       *host_shape,
                      const long long *host_strides_b) {
    if (n_elems <= 0) return 0;                   /* nothing to copy */
    if (ndim <= 0 || ndim > STRIDED_COPY_MAX_NDIM) return -1;
    if (elsize <= 0) return -1;

    struct StridedCopyDims dims;
    dims.ndim   = ndim;
    dims.elsize = elsize;
    for (int i = 0; i < ndim; i++) {
        dims.shape[i]     = host_shape[i];
        dims.strides_b[i] = host_strides_b[i];
    }
    /* Zero-pad the unused tail so the struct never has indeterminate bytes
       (the kernel only reads the first `ndim` entries, but valgrind-style
       tooling on the host side prefers fully initialised payloads). */
    for (int i = ndim; i < STRIDED_COPY_MAX_NDIM; i++) {
        dims.shape[i]     = 1;
        dims.strides_b[i] = 0;
    }

    const int blockSize = 256;
    long long blocks_ll = (n_elems + blockSize - 1) / blockSize;
    /* CUDA's grid x-dimension limit is 2^31 - 1 on all supported archs.
       Realistic NDArrays never come close, but guard anyway. */
    if (blocks_ll > 2147483647LL) return -1;
    int numBlocks = (int)blocks_ll;

    tcuda_strided_copy_kernel<<<numBlocks, blockSize>>>(
        dst_gpu, src_gpu, n_elems, dims);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

/* ───────────────── Normal sampler ──────────────────────────────────────── */

/**
 * @brief Derive a fresh PRNG seed for each `cuda_normal_*` call.
 *
 * Combines wall-clock seconds with a monotonically-increasing 64-bit
 * counter so successive calls in the same second still produce
 * statistically-independent streams. The previous implementation pinned
 * the seed to `1234ULL` which made repeated calls within one process
 * generate identical sample sequences — a real entropy bug.
 *
 * @return New 64-bit seed.
 */
static unsigned long long cuda_normal_next_seed(void) {
    static unsigned long long counter = 0;
    unsigned long long t = (unsigned long long)time(NULL);
    /* The 17-bit shift gives each call a wide jump in the seed even when
       the wall-clock didn't advance; the xor lets `time(NULL) == 0` (an
       edge case during early init) still produce non-zero seeds. */
    return (t << 32) ^ (counter++ << 17) ^ 0x9e3779b97f4a7c15ULL;
}

/**
 * @brief curandStatus → return-or-throw helper.
 *
 * cuRAND failure modes here are: out-of-memory, invalid handle, no
 * device. None are recoverable inside the wrapper so we surface them as
 * a catchable PHP Error and leave the destination buffer in whatever
 * state cuRAND left it (caller will `NDArray_FREE` and propagate
 * `return NULL`).
 *
 * @param[in] st curandStatus_t.
 * @return 1 on CURAND_STATUS_SUCCESS, 0 on every failure (Error in flight).
 */
static int cuda_normal_check(curandStatus_t st) {
    if (st == CURAND_STATUS_SUCCESS) return 1;
    /* Don't throw a PHP error from libcudart-linked code; just signal
       failure. Callers translate to an Error in PHP-frame functions. */
    return 0;
}

void cuda_normal_f32(float *d_data, long n, float mean, float stddev) {
    if (d_data == NULL || n <= 0) return;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        return;
    }
    curandSetPseudoRandomGeneratorSeed(gen, cuda_normal_next_seed());

    if ((n & 1) == 0) {
        /* Even size — cuRAND can write directly into the destination. */
        cuda_normal_check(curandGenerateNormal(gen, d_data, (size_t)n,
                                                mean, stddev));
    } else {
        /* Odd size — cuRAND requires even. Write `n + 1` into a transient
           pad buffer (vmalloc so NDARRAY_VCHECK sees it), then copy
           `n` floats into the destination. The pad buffer is freed before
           the wrapper returns. */
        float *pad = NULL;
        vmalloc((void **)&pad, (unsigned int)(sizeof(float) * (size_t)(n + 1)));
        if (pad != NULL) {
            cuda_normal_check(curandGenerateNormal(gen, pad, (size_t)(n + 1),
                                                    mean, stddev));
            cudaMemcpy(d_data, pad, sizeof(float) * (size_t)n,
                       cudaMemcpyDeviceToDevice);
            vfree(pad);
        }
    }
    curandDestroyGenerator(gen);
}

void cuda_normal_f64(double *d_data, long n, double mean, double stddev) {
    if (d_data == NULL || n <= 0) return;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        return;
    }
    curandSetPseudoRandomGeneratorSeed(gen, cuda_normal_next_seed());

    if ((n & 1) == 0) {
        cuda_normal_check(curandGenerateNormalDouble(gen, d_data, (size_t)n,
                                                      mean, stddev));
    } else {
        /* Odd size — same pad-and-copy trick as f32, but through vmalloc
           so NDARRAY_VCHECK can balance the allocation against vfree. */
        double *pad = NULL;
        vmalloc((void **)&pad, (unsigned int)(sizeof(double) * (size_t)(n + 1)));
        if (pad != NULL) {
            cuda_normal_check(curandGenerateNormalDouble(gen, pad,
                                                          (size_t)(n + 1),
                                                          mean, stddev));
            cudaMemcpy(d_data, pad, sizeof(double) * (size_t)n,
                       cudaMemcpyDeviceToDevice);
            vfree(pad);
        }
    }
    curandDestroyGenerator(gen);
}

void cuda_truncated_normal_f32(float *d_data, long n, float loc, float scale) {
    if (d_data == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    truncatedNormalKernelF32<<<blocks, block>>>(d_data, (int)n, loc, scale,
                                                  cuda_normal_next_seed());
}

void cuda_truncated_normal_f64(double *d_data, long n, double loc, double scale) {
    if (d_data == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    truncatedNormalKernelF64<<<blocks, block>>>(d_data, (int)n, loc, scale,
                                                  cuda_normal_next_seed());
}

/**
 * @brief Per-thread DD affine kernel: `dst[i] = loc + scale * z[i]` in dd.
 *
 * Reads one standard-normal double `z[i]`, computes the affine transform
 * in true double-double arithmetic on device, and stores the result at
 * `dst[2i..2i+1]`. The transform is intentionally done with full DD
 * precision so a caller's fp128 `loc`/`scale` survive the trip through
 * VRAM intact; the only precision loss along the pipeline is the
 * standard-normal sample `z`, which is inherently 53-bit (cuRAND's
 * `curandGenerateNormalDouble` produces fp64 samples).
 */
__global__ void cuda_normal_dd_affine_kernel(const double *z, double *dst,
                                              long n,
                                              double loc_hi, double loc_lo,
                                              double scale_hi, double scale_lo) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dd_real zdd    = dd_make(z[i], 0.0);
        dd_real scale  = dd_make(scale_hi, scale_lo);
        dd_real loc    = dd_make(loc_hi, loc_lo);
        dd_real prod   = dd_mul(zdd, scale);
        dd_real result = dd_add(loc, prod);
        dst[2*i]     = result.hi;
        dst[2*i + 1] = result.lo;
    }
}

void cuda_normal_dd_affine(const double *z, double *dst, long n,
                           double loc_hi, double loc_lo,
                           double scale_hi, double scale_lo) {
    if (z == NULL || dst == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_normal_dd_affine_kernel<<<blocks, block>>>(z, dst, n,
                                                      loc_hi, loc_lo,
                                                      scale_hi, scale_lo);
}

/* ───────────────── Uniform sampler ─────────────────────────────────────── */

/**
 * @brief Per-thread affine for the float32 uniform path.
 *
 * cuRAND's `curandGenerateUniform` returns values in `(0, 1]`. To match
 * numpy's `[low, high)` contract we reflect with `1 - u`, mapping to
 * `[0, 1)`, then evaluate `low + (1 - u) * (high - low)` so the closed
 * endpoint sits at `low` (and the open endpoint at `high`). All math is
 * done in single precision to keep the float32 path's quantisation
 * deterministic.
 *
 * @param[in,out] data Length-@p n buffer of `(0, 1]` samples on entry; on
 *                     return each slot holds `low + (1 - u_in) * (high - low)`.
 * @param[in]     n    Element count.
 * @param[in]     low  Lower bound (inclusive).
 * @param[in]     high Upper bound (exclusive).
 */
__global__ void cuda_uniform_affine_kernel_f32(float *data, long n,
                                                 float low, float high) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = low + (1.0f - data[i]) * (high - low);
    }
}

/**
 * @brief Float64 companion of `cuda_uniform_affine_kernel_f32`.
 *
 * Identical reflection / affine, evaluated in double precision so the
 * fp64 path keeps full 53-bit mantissa precision across the whole
 * `[low, high)` range.
 *
 * @param[in,out] data Length-@p n buffer of `(0, 1]` samples on entry.
 * @param[in]     n    Element count.
 * @param[in]     low  Lower bound (inclusive).
 * @param[in]     high Upper bound (exclusive).
 */
__global__ void cuda_uniform_affine_kernel_f64(double *data, long n,
                                                 double low, double high) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = low + (1.0 - data[i]) * (high - low);
    }
}

void cuda_uniform_f32(float *d_data, long n, float low, float high) {
    if (d_data == NULL || n <= 0) return;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        return;
    }
    curandSetPseudoRandomGeneratorSeed(gen, cuda_normal_next_seed());
    /* `curandGenerateUniform` has no parity restriction (unlike
       `curandGenerateNormal`) so a single in-place call into the
       destination is enough; no pad buffer needed. */
    cuda_normal_check(curandGenerateUniform(gen, d_data, (size_t)n));
    curandDestroyGenerator(gen);

    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_uniform_affine_kernel_f32<<<blocks, block>>>(d_data, n, low, high);
}

void cuda_uniform_f64(double *d_data, long n, double low, double high) {
    if (d_data == NULL || n <= 0) return;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        return;
    }
    curandSetPseudoRandomGeneratorSeed(gen, cuda_normal_next_seed());
    cuda_normal_check(curandGenerateUniformDouble(gen, d_data, (size_t)n));
    curandDestroyGenerator(gen);

    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_uniform_affine_kernel_f64<<<blocks, block>>>(d_data, n, low, high);
}

/**
 * @brief Per-thread DD affine kernel for the float128 uniform GPU path.
 *
 * Reads one `[0, 1)` double `u[i]` (the reflection from `(0, 1]` is
 * applied upstream by `cuda_uniform_f64` with `low=0, high=1`), and
 * computes `low + u * range` in true double-double arithmetic on
 * device. `range = high - low` is supplied as a DD pair computed on
 * the host so the kernel itself does not need to perform a DD
 * subtraction. The result is stored at `dst[2i..2i+1]`. The only
 * precision loss along the pipeline is the underlying uniform sample,
 * which is inherently 53-bit (cuRAND's `curandGenerateUniformDouble`).
 *
 * @param[in]  u         Length-@p n GPU buffer of `[0, 1)` doubles.
 * @param[out] dst       Length-`2*n` GPU buffer of interleaved (hi, lo) pairs.
 * @param[in]  n         Element count.
 * @param[in]  low_hi    DD high word of the lower bound.
 * @param[in]  low_lo    DD low word of the lower bound.
 * @param[in]  range_hi  DD high word of `(high - low)`.
 * @param[in]  range_lo  DD low word of `(high - low)`.
 */
__global__ void cuda_uniform_dd_affine_kernel(const double *u, double *dst,
                                                long n,
                                                double low_hi, double low_lo,
                                                double range_hi, double range_lo) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dd_real udd    = dd_make(u[i], 0.0);
        dd_real range  = dd_make(range_hi, range_lo);
        dd_real low    = dd_make(low_hi, low_lo);
        dd_real prod   = dd_mul(udd, range);
        dd_real result = dd_add(low, prod);
        dst[2*i]     = result.hi;
        dst[2*i + 1] = result.lo;
    }
}

void cuda_uniform_dd_affine(const double *u, double *dst, long n,
                             double low_hi, double low_lo,
                             double range_hi, double range_lo) {
    if (u == NULL || dst == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_uniform_dd_affine_kernel<<<blocks, block>>>(u, dst, n,
                                                       low_hi, low_lo,
                                                       range_hi, range_lo);
}

/* ───────────────── uint64 affine kernels ──────────────────────────────── */

/**
 * @brief Per-thread normal/truncated-normal affine kernel for the uint64
 *        GPU path.
 *
 * Reads one (possibly truncated) standard-normal double `z[i]`,
 * evaluates `delta_s = (int64_t)(scaled * z[i])` (signed so negative-z
 * samples subtract from `loc`), and writes `loc + (uint64_t)delta_s` to
 * `dst[i]`. The signed→unsigned cast wraps modulo 2^64 — well-defined
 * in C/C++ for unsigned destinations and matches the CPU filler's
 * arithmetic. Used for both `NDArray_Normal` and `NDArray_TruncatedNormal`
 * (the caller picks the source distribution by which cuRAND fill
 * populates @p z).
 *
 * @param[in]  z      Length-@p n GPU buffer of standard-normal (or
 *                    truncated standard-normal) doubles.
 * @param[out] dst    Length-@p n GPU uint64 buffer.
 * @param[in]  n      Element count.
 * @param[in]  loc    Distribution mean (uint64).
 * @param[in]  scaled Distribution stddev coerced to double.
 */
__global__ void cuda_normal_u64_affine_kernel(const double *z,
                                                unsigned long long *dst,
                                                long n,
                                                unsigned long long loc,
                                                double scaled) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long delta_s = (long long)(scaled * z[i]);
        dst[i] = loc + (unsigned long long)delta_s;
    }
}

void cuda_normal_u64_affine(const double *z, unsigned long long *dst, long n,
                             unsigned long long loc, double scaled) {
    if (z == NULL || dst == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_normal_u64_affine_kernel<<<blocks, block>>>(z, dst, n, loc, scaled);
}

/**
 * @brief Per-thread uniform affine kernel for the uint64 GPU path.
 *
 * Reads one `[0, 1)` double `u[i]` (callers pre-reflect via
 * `cuda_uniform_f64(u, n, 0.0, 1.0)`) and writes
 * `low + (uint64_t)(widthd * u[i])` to `dst[i]`. The width is supplied
 * as a `double` because the cast `(double)(high - low)` happens once on
 * the host — for widths past 2^53 this is the same precision floor
 * the CPU filler hits (documented invariant). The unsigned add wraps
 * modulo 2^64.
 *
 * @param[in]  u      Length-@p n GPU buffer of `[0, 1)` doubles.
 * @param[out] dst    Length-@p n GPU uint64 buffer.
 * @param[in]  n      Element count.
 * @param[in]  low    Lower bound (uint64).
 * @param[in]  widthd `(double)(high - low)`.
 */
__global__ void cuda_uniform_u64_affine_kernel(const double *u,
                                                 unsigned long long *dst,
                                                 long n,
                                                 unsigned long long low,
                                                 double widthd) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = low + (unsigned long long)(widthd * u[i]);
    }
}

void cuda_uniform_u64_affine(const double *u, unsigned long long *dst, long n,
                              unsigned long long low, double widthd) {
    if (u == NULL || dst == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_uniform_u64_affine_kernel<<<blocks, block>>>(u, dst, n, low, widthd);
}

/* ───────────────── Poisson sampler ─────────────────────────────────────── */

int cuda_poisson_u32(unsigned int *d_data, long n, double lam) {
    if (d_data == NULL || n <= 0) return 1;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        return 0;
    }
    curandSetPseudoRandomGeneratorSeed(gen, cuda_normal_next_seed());
    /* `curandGeneratePoisson` writes `n` uint32 samples directly into
       @p d_data — no parity restriction, no scratch buffer required.
       Internally cuRAND picks between rejection-from-normal and PTRS
       depending on @p lam. cuRAND returns `CURAND_STATUS_OUT_OF_RANGE`
       (or a different non-success code) when @p lam exceeds the
       generator's supported range; we surface that failure to the
       caller so a clear error can be raised at the PHP boundary
       instead of returning a silently-zero buffer. */
    int ok = cuda_normal_check(curandGeneratePoisson(gen, d_data,
                                                      (size_t)n, lam));
    curandDestroyGenerator(gen);
    return ok;
}

/**
 * @brief Per-thread widening kernel: write each u32 sample as a DD pair
 *        with `lo = 0.0`.
 *
 * The destination layout is the interleaved (hi, lo) format the rest of
 * the fp128 GPU pipeline uses: `dst[2i] = (double)src[i]`,
 * `dst[2i + 1] = 0.0`. Every uint32 fits exactly in fp64's 53-bit
 * mantissa, so the high word carries the integer count without loss
 * and the low word is identically zero.
 *
 * @param[in]  src Length-@p n GPU buffer of uint32 Poisson samples.
 * @param[out] dst Length-`2*n` GPU buffer of (hi, lo) DD pairs.
 * @param[in]  n   Element count.
 */
__global__ void cuda_cast_u32_to_dd_kernel(const unsigned int *src,
                                             double *dst, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[2*i]     = (double)src[i];
        dst[2*i + 1] = 0.0;
    }
}

void cuda_cast_u32_to_dd(const unsigned int *src, double *dst, long n) {
    if (src == NULL || dst == NULL || n <= 0) return;
    int block = 256;
    long blocks_ll = (n + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_cast_u32_to_dd_kernel<<<blocks, block>>>(src, dst, n);
}

/* ───────────────── Binomial sampler ────────────────────────────────────── */

/**
 * @brief Per-thread direct-Bernoulli kernel for the Binomial sampler.
 *
 * Each thread owns one output slot, initialises its own cuRAND state
 * from `(seed, idx)`, runs @p n independent Bernoulli trials with
 * success probability @p p, and writes the success count as a uint32.
 * `curand_uniform` returns `(0, 1]`; we reflect to `[0, 1)` via
 * `1 - u` so the comparison `u < p` honours the closed-open convention
 * (matches the CPU sampler).
 *
 * Cost is `O(n)` per element — fine for small to moderate @p n
 * (< ~10^4); for very large @p n the call is still correct but a more
 * sophisticated algorithm (BTPE) would be faster. The legacy CPU
 * implementation also used this direct method.
 *
 * @param[out] dst  Destination GPU buffer of @p total uint32s.
 * @param[in]  total Element count.
 * @param[in]  n     Number of Bernoulli trials per sample.
 * @param[in]  p     Per-trial success probability in `[0, 1]`.
 * @param[in]  seed  Per-call seed (`cuda_normal_next_seed`).
 */
__global__ void cuda_binomial_kernel(unsigned int *dst, long total,
                                       int n, float p,
                                       unsigned long long seed) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    curandState_t state;
    curand_init(seed, (unsigned long long)idx, 0, &state);
    unsigned int successes = 0;
    for (int j = 0; j < n; j++) {
        float u = 1.0f - curand_uniform(&state);  /* (0, 1] → [0, 1) */
        if (u < p) successes++;
    }
    dst[idx] = successes;
}

void cuda_binomial_u32(unsigned int *d_data, long total, int n, float p) {
    if (d_data == NULL || total <= 0) return;
    /* n == 0 is degenerate (every sample is 0); the kernel handles it
       correctly (the per-thread loop is empty) but the upstream
       dispatcher short-circuits with cudaMemset before reaching us. */
    int block = 256;
    long blocks_ll = (total + block - 1) / block;
    if (blocks_ll > 2147483647LL) return;
    int blocks = (int)blocks_ll;
    cuda_binomial_kernel<<<blocks, block>>>(d_data, total, n, p,
                                              cuda_normal_next_seed());
}

}