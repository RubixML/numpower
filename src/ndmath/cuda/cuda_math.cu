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

__global__ void truncatedNormalKernel(float* d_data, int size, double loc, double scale, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;

    if (idx < size) {
        curand_init(seed, idx, 0, &state);
        float z;
        do {
            z = curand_normal(&state) * scale + loc;
        } while (z < (loc - 2.0 * scale) || z > (loc + 2.0 * scale));
        d_data[idx] = z;
    }
}

void cuda_truncated_normal(float* h_data, int size, double loc, double scale) {
    // Определение параметров сетки и блоков
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Вызов ядра CUDA
    truncatedNormalKernel<<<blocksPerGrid, threadsPerBlock>>>(h_data, size, loc, scale, 1234ULL);
    cudaDeviceSynchronize();
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
void arctan2FloatKernel(float* d_array, float* d_array2, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = atan2f(d_array[index], d_array2[index]);
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
void expm1FloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = expm1f(d_array[index]);
    }
}

__global__
void expFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = expf(d_array[index]);
    }
}

__global__
void sqrtFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sqrtf(d_array[index]);
    }
}

__global__
void logFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = logf(d_array[index]);
    }
}

__global__
void logbFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = logbf(d_array[index]);
    }
}

__global__
void log2FloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = log2f(d_array[index]);
    }
}

__global__
void log1pFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = log1pf(d_array[index]);
    }
}

__global__
void log10FloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = log10f(d_array[index]);
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

__device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
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

__global__ void
max_reduce_naive(float * result, float * data, int size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load data into shared memory
    if (i < size) {
        sharedData[tid] = data[i];
        if (i + blockDim.x < size) {
            sharedData[tid + blockDim.x] = data[i + blockDim.x];
        } else {
            // If the last block has an odd number of elements, duplicate the last element
            sharedData[tid + blockDim.x] = data[size - 1];
        }
    } else {
        // If the last block has an odd number of elements, duplicate the last element
        sharedData[tid] = data[size - 1];
        sharedData[tid + blockDim.x] = data[size - 1];
    }

    __syncthreads();

    // Parallel reduction within the warp to find the maximum value
    float maxVal = sharedData[tid];
    maxVal = warpReduceMax(maxVal);

    // The maximum value within the warp is in maxVal
    if ((tid & (warpSize - 1)) == 0) {
        sharedData[tid / warpSize] = maxVal;
    }

    __syncthreads();

    // Further reduction using one thread per warp
    if (tid < blockDim.x / warpSize) {
        maxVal = sharedData[tid];
        maxVal = warpReduceMax(maxVal);
        if (tid == 0) {
            atomicMaxFloat(result, maxVal);
        }
    }
}

__device__ float warpReduceMin(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
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

__global__ void
min_reduce_naive(float * result, float * data, int size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load data into shared memory
    if (i < size) {
        sharedData[tid] = data[i];
        if (i + blockDim.x < size) {
            sharedData[tid + blockDim.x] = data[i + blockDim.x];
        } else {
            // If the last block has an odd number of elements, duplicate the last element
            sharedData[tid + blockDim.x] = data[size - 1];
        }
    } else {
        // If the last block has an odd number of elements, duplicate the last element
        sharedData[tid] = data[size - 1];
        sharedData[tid + blockDim.x] = data[size - 1];
    }

    __syncthreads();

    // Parallel reduction within the warp to find the minimum value
    float minVal = sharedData[tid];
    minVal = warpReduceMin(minVal);

    // The minimum value within the warp is in minVal
    if ((tid & (warpSize - 1)) == 0) {
        sharedData[tid / warpSize] = minVal;
    }

    __syncthreads();

    // Further reduction using one thread per warp
    if (tid < blockDim.x / warpSize) {
        minVal = sharedData[tid];
        minVal = warpReduceMin(minVal);
        if (tid == 0) {
            atomicMinFloat(result, minVal);
        }
    }
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

    // write result for this block to global mem
    if (tid == 0) atomicAdd(result, sdata[0]);
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

__global__
void fill_float_kernel(float* array, int n, float value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        array[idx] = value;
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
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
        cublasCreate(&cublasH);
        cublasSetStream(cublasH, stream);

        int* d_Ipiv; // pivot array
        int* d_info;  // info on success or failure
        float* d_U; // U matrix of LU decomposition

        CHECK_CUDA(cudaMalloc(&d_Ipiv, N*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_U, N*N*sizeof(float)));

        // copy A to U as cusolverDnSgetrf works in place
        CHECK_CUDA(cudaMemcpy(d_U, d_A, N*N*sizeof(float), cudaMemcpyDeviceToDevice));

        // LU decompose
        cusolver_status = cusolverDnSgetrf(cusolverH, N, N, d_U, N, NULL, d_Ipiv, d_info);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            // handle error
            printf("LU decomposition failed\n");
            exit(1);
        }

        // Find determinant by product of diagonal elements
        float det = 1.0f;
        for (int i = 0; i < N; i++) {
            float elem;
            CHECK_CUDA(cudaMemcpy(&elem, d_U + i * N + i, sizeof(float), cudaMemcpyDeviceToHost));
            // Check for potential overflow
            if (fabsf(elem) > FLT_MAX / fabsf(det)) {
                // Handle overflow here, e.g., return a special value or throw an error
                printf("Overflow detected in det\n");
                exit(1);
            }
            if (!isnan(elem) && !isinf(elem)) {
                det *= elem;
            }
        }

        // Analyze pivot array to calculate number of permutations
        int* h_Ipiv = new int[N];
        CHECK_CUDA(cudaMemcpy(h_Ipiv, d_Ipiv, N*sizeof(int), cudaMemcpyDeviceToHost));

        int numPermutations = 0;
        for(int i = 0; i < N; i++) {
            if(i+1 != h_Ipiv[i]) numPermutations++;
        }

        if(numPermutations % 2 != 0) det = -det;

        // Cleanup
        if (d_U) cudaFree(d_U);
        if (d_Ipiv) cudaFree(d_Ipiv);
        if (d_info) cudaFree(d_info);
        if (cublasH) cublasDestroy(cublasH);
        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (stream) cudaStreamDestroy(stream);

        CHECK_CUDA(cudaMemcpy(result, &det, sizeof(float), cudaMemcpyHostToDevice));
        return 1;
    }

    void
    cuda_fill_float(float *a, float value, int n) {
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        fill_float_kernel<<<gridSize, blockSize>>>(a, n, value);
        cudaDeviceSynchronize();
    }

    void
    cuda_sum_float(int nblocks, float *a, float *rtn, int nelements) {
        float *d_sum;
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize * 2 - 1) / (blockSize * 2);  // Number of blocks in the grid.
        cudaMalloc((void **) &d_sum, sizeof(float));

        cudaMemcpy(d_sum, rtn, sizeof(float), cudaMemcpyHostToDevice);
        array_sum_float<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(a, d_sum, nelements);
        cudaMemcpy(rtn, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    void
    cuda_prod_float(int nblocks, float *a, float *rtn, int nelements) {
        float *d_prod;
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize * 2 - 1) / (blockSize * 2);  // Number of blocks in the grid.
        cudaMalloc((void **) &d_prod, sizeof(float));

        cudaMemcpy(d_prod, rtn, sizeof(float), cudaMemcpyHostToDevice);
        array_prod_float<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(a, d_prod, nelements);
        cudaMemcpy(rtn, d_prod, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    int
    cuda_svd_float(float *d_A, float *d_U, float *d_V, float *d_S, int m, int n) {
        cusolverDnHandle_t cusolverH = NULL;  // cuSOLVER handle
        cudaStream_t stream = NULL;  // CUDA stream
        gesvdjInfo_t gesvdj_params = NULL;  // configuration of gesvdj
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
        CHECK_CUSOLVER(cusolverDnCreateGesvdjInfo(&gesvdj_params));

        // Set desired configuration of gesvdj
        CHECK_CUSOLVER(cusolverDnXgesvdjSetTolerance(
                gesvdj_params,
                1.e-7));
        CHECK_CUSOLVER(cusolverDnXgesvdjSetMaxSweeps(
                gesvdj_params,
                15));

        // Perform SVD
        // Note: This is just a skeleton code. Please handle CUDA errors appropriately
        int* devInfo = NULL;  // info on gesvdj convergence
        CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));
        int lwork = 0;
        float *d_work = NULL;
        CHECK_CUSOLVER(cusolverDnSgesvdj_bufferSize(
                cusolverH,
                CUSOLVER_EIG_MODE_VECTOR,  // compute eigenvectors
                0,  // number of singular values to compute, 0 for all
                m,
                n,
                d_A,
                m,  // leading dimension of A
                d_S,
                d_U,
                m,  // leading dimension of U
                d_V,
                n,  // leading dimension of V
                &lwork,
                gesvdj_params));

        CHECK_CUDA(cudaMalloc((void**)&d_work , sizeof(float) * lwork));
        CHECK_CUSOLVER(cusolverDnSgesvdj(
                cusolverH,
                CUSOLVER_EIG_MODE_VECTOR,  // compute eigenvectors
                0,  // number of singular values to compute, 0 for all
                m,
                n,
                d_A,
                m,  // leading dimension of A
                d_S,
                d_U,
                m,  // leading dimension of U
                d_V,
                n,  // leading dimension of V
                d_work,
                lwork,
                devInfo,
                gesvdj_params));

        // Synchronize to ensure computation is finished
        CHECK_CUDA(cudaDeviceSynchronize());
        if (devInfo) CHECK_CUDA(cudaFree(devInfo));
        if (cusolverH) CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
        if (stream) CHECK_CUDA(cudaStreamDestroy(stream));
        if (gesvdj_params) CHECK_CUSOLVER(cusolverDnDestroyGesvdjInfo(gesvdj_params));

        return 1;
    }

    float
    cuda_max_float(float *a, int nelements) {
        int size = nelements;
        float *d_out;
        int current_size = size;
        float *d_current_in = a;
        // Launch the CUDA kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + (2 * threadsPerBlock) - 1) / (2 * threadsPerBlock);
        cudaMalloc((void**)&d_out, sizeof(float));
        max_reduce_naive<<<blocksPerGrid, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(d_out, d_current_in, current_size);
        float max_value;
        cudaMemcpy(&max_value, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        return max_value;
    }

    float
    cuda_min_float(float *a, int nelements) {
        int size = nelements;
        float *d_out;
        int current_size = size;
        float *d_current_in = a;
        // Launch the CUDA kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + (2 * threadsPerBlock) - 1) / (2 * threadsPerBlock);
        cudaMalloc((void**)&d_out, sizeof(float));
        min_reduce_naive<<<blocksPerGrid, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(d_out, d_current_in, current_size);
        float min_value;
        cudaMemcpy(&min_value, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        return min_value;
    }

    int
    cuda_equal_float(int nblocks, float *a, float *b, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int result = 1;
        int *d_equal;
        // Allocate GPU memory for the result
        cudaMalloc(&d_equal, sizeof(int));
        cudaMemcpy(d_equal, &result, sizeof(int), cudaMemcpyHostToDevice);
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        array_equals_float<<<numBlocks, blockSize>>>(a, b, d_equal, nelements);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_equal, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_equal);
        return result;
    }

    void
    cuda_pow_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        pow_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_mod_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        fmodf_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_multiply_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        multiply_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_divide_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        divide_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_subtract_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        subtract_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_add_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        add_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        logFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_logb(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        logbFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log2(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        log2FloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log1p(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        log1pFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log10(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        log10FloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sqrt(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sqrtFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_exp(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        expFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_abs(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        absFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_expm1(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        expm1FloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sin(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sinFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_cos(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        cosFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_tan(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        tanFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arcsin(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arcsinFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arctan(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctanFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arctan2(int nblocks, float *d_array, float *y_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctan2FloatKernel<<<numBlocks, blockSize>>>(d_array, y_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arccos(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arccosFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_radians(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        radiansFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_degrees(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        degreesFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sinh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sinhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_cosh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        coshFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_tanh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        tanhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arcsinh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arcsinhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_transpose(int tiledim, int blockrows, const float *d_in, float *d_out, int width, int height) {

        dim3 grid(16, 16);
        dim3 block(16, 16);
        transposeCoalesced<<<grid, block>>>(d_in, height, width, d_out);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arccosh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arccoshFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arctanh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctanhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_rint(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        rintFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_fix(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        fixFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_ceil(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        ceilFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_round(int nblocks, float *d_array, float decimals) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        roundToDecimalsFloatKernel<<<numBlocks, blockSize>>>(d_array, (int)decimals, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_floor(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        floorFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_trunc(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        truncFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sinc(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sincFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_calculate_outer_product(int m, int n, float *a_array, float *b_array, float *r_array) {
        dim3 blockSize(16, 16);  // Number of threads per block. This is a typical choice.
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        calculateOuterProductFloat<<<gridSize, blockSize>>>(a_array, b_array, m, n, r_array);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_negate(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        negateFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_positive(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        positiveFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_reciprocal(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        reciprocalFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sign(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        signFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_clip(int nblocks, float *d_array, float minVal, float maxVal) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        clipFloatKernel<<<numBlocks, blockSize>>>(d_array, minVal, maxVal, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_multiply_matrix_vector(int nblocks, float *a_array, float *b_array, float *result, int rows, int cols) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        matrixVectorMultiplyFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, rows, cols);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_compare_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_compare_not_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysNotEqualFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_compare_greater(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysGreaterFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_compare_greater_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysGreaterEqualFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
        cudaDeviceSynchronize();
    }

    float
    cuda_float_median_float(int nblocks, float *a_array, int n) {
        const int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        float* d_medians;
        cudaMalloc((void**)&d_medians, blocksPerGrid * sizeof(float));

        findMedianKernelFloat<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(a_array, n, d_medians);

        // Perform a final reduction to find the overall median
        while (blocksPerGrid > 1)
        {
            int newBlocks = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
            findMedianKernelFloat<<<newBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_medians, blocksPerGrid, d_medians);
            blocksPerGrid = newBlocks;
        }

        float median;
        cudaMemcpy(&median, d_medians, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_medians);

        return median;
    }

    void
    cuda_float_compare_less(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysLessFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_compare_less_equal(int nblocks, float *a_array, float *b_array, float *result, int n) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        compareArraysLessEqualFloatKernel<<<numBlocks, blockSize>>>(a_array, b_array, result, n);
        cudaDeviceSynchronize();
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
        op(NDArray_NUMELEMENTS(rtn), NDArray_FDATA(rtn));
        return rtn;
    }

    NDArray*
    NDArrayMathGPU_ElementWise1F(NDArray* ndarray, ElementWiseFloatGPUOperation1F op, float val1) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_FDATA(rtn), val1);
        return rtn;
    }

    NDArray*
    NDArrayMathGPU_ElementWise1N(NDArray* ndarray, ElementWiseFloatGPUOperation1N op, NDArray* val1) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_FDATA(rtn), NDArray_FDATA(val1));
        return rtn;
    }

    NDArray*
    NDArrayMathGPU_ElementWise2F(NDArray* ndarray, ElementWiseFloatGPUOperation2F op, float val1, float val2) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_FDATA(rtn), val1, val2);
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

    int
    cuda_matrix_float_l2norm(float *target, float *rtn, int rows, int cols) {
        cusolverDnHandle_t handle;
        CHECK_CUSOLVER(cusolverDnCreate(&handle));

        // Calculate workspace size for SVD
        int work_size;
        CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(handle, rows, cols, &work_size));

        // Allocate workspace
        float* d_work;
        CHECK_CUDA(cudaMalloc((void**)&d_work, work_size));

        // Allocate singular values
        float* d_singular_values;
        CHECK_CUDA(cudaMalloc((void**)&d_singular_values, cols * sizeof(float)));

        // Perform SVD
        CHECK_CUSOLVER(cusolverDnSgesvd(handle, 'N', 'N', rows, cols, target, rows, d_singular_values, NULL, rows, NULL, cols, d_work, work_size, NULL, NULL));

        CHECK_CUDA(cudaMemcpy(rtn, d_singular_values, sizeof(float), cudaMemcpyDeviceToDevice));

        // Cleanup
        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaFree(d_singular_values));
        CHECK_CUSOLVER(cusolverDnDestroy(handle));
        return 0;
    }

    void cuda_matrix_float_inverse(float* matrix, int n) {
        cusolverDnHandle_t cusolverH;
        cusolverDnCreate(&cusolverH);

        float* d_matrix = matrix;

        int* d_info;
        vmalloc((void**)&d_info, sizeof(int));

        int lwork;
        cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_matrix, n, &lwork);

        float* d_work;
        vmalloc((void**)&d_work, lwork * sizeof(float));

        int* d_pivot;
        vmalloc((void**)&d_pivot, n * sizeof(int));

        cusolverDnSgetrf(cusolverH, n, n, d_matrix, n, d_work, d_pivot, d_info);

        float* d_identity;
        vmalloc((void**)&d_identity, n * n * sizeof(float));
        cudaMemset(d_identity, 0, n * n * sizeof(float));
        float onef = 1.0f;
        for (int i = 0; i < n; ++i)
            cudaMemcpy(d_identity + i * n + i, &onef, sizeof(float), cudaMemcpyHostToDevice);

        cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, d_matrix, n, d_pivot, d_identity, n, d_info);

        cudaMemcpy(matrix, d_identity, n * n * sizeof(float), cudaMemcpyDeviceToHost);

        vfree(d_info);
        vfree(d_work);
        vfree(d_pivot);
        vfree(d_identity);

        cusolverDnDestroy(cusolverH);
    }

    void cuda_matrix_eig_float(float* d_matrix, int n, float* d_eigvalues) {
        cusolverDnHandle_t handle;
        cusolverStatus_t status;

        int* d_info;  // info on success or failure

        cudaMalloc(&d_info, sizeof(int));
        // Create cuSOLVER handle
        status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            printf("CUSOLVER initialization failed.\n");
            return;
        }

        // Compute workspace size
        int lwork;
        status = cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_matrix, n, d_eigvalues, &lwork);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            printf("CUSOLVER workspace size computation failed.\n");
            return;
        }

        // Allocate workspace on the device
        float* d_work;
        cudaMalloc((void**)&d_work, lwork * sizeof(float));

        // Compute eigenvalues and right eigenvectors
        status = cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_matrix, n, d_eigvalues, d_work, lwork, d_info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            printf("CUSOLVER eigenvectors computation failed.\n");
            return;
        }
        cudaFree(d_work);
        cudaFree(d_info);
    }


}