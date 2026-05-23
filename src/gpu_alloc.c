#ifndef _MSC_VER
#include "../config.h"
#endif
#include <Zend/zend.h>

#ifdef HAVE_CUBLAS
#include "gpu_alloc.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "buffer.h"

/**
 * @brief Allocate a GPU buffer and track it in the VCHECK counter.
 *
 * Size-0 requests bypass `cudaMalloc` and leave `*target` as `NULL`.
 * The accounting counter is only incremented when an actual device
 * allocation occurs, so `NDArray_FREE` can no-op safely when an empty
 * NDArray (shape with a zero dim) is destroyed and the counter still
 * balances under `NDARRAY_VCHECK=1`.
 *
 * @param[out] target receives the device pointer (NULL on size==0).
 * @param[in]  size   bytes to allocate; 0 is a valid no-op.
 */
void
vmalloc(void** target, unsigned int size) {
    if (size == 0) {
        *target = NULL;
        return;
    }
    cublasStatus_t stat = cudaMalloc(target, size);
    if (stat != cudaSuccess) {
        zend_throw_error(NULL, "device memory allocation failed");
        return;
    }
    MAIN_MEM_STACK.totalGPUAllocated++;
}

/**
 * @brief Free a GPU buffer previously returned by `vmalloc`.
 *
 * `vfree(NULL)` is a safe no-op — it does not touch the counter so the
 * size-0 allocations skipped by `vmalloc` stay balanced.
 *
 * @param[in] target device pointer to free; NULL is accepted.
 */

void
vmemcpyd2d(char* target, char* dst, unsigned int size) {
    cudaMemcpy(dst, target, size, cudaMemcpyDeviceToDevice);
}

void
vmemcpyh2d(char* target, char* dst, unsigned int size) {
    cudaMemcpy(dst, target, size, cudaMemcpyHostToDevice);
}

void
vfree(void* target) {
    if (target == NULL) {
        return;
    }
    MAIN_MEM_STACK.totalGPUAllocated--;
    cudaFree(target);
}

/**
 * @brief Broadcast @p value_host across @p n elements of @p dst on the device.
 *
 * Doubling-prefix implementation: seed dst[0] from the host, then repeatedly
 * cudaMemcpy the populated prefix onto the unwritten tail. The work is
 * D2D except for the single H2D seed of @p elsize bytes, so host RAM
 * usage stays at whatever the caller passes — independent of @p n.
 *
 * @see gpu_alloc.h::cuda_fill_bytes for the public contract.
 */
void
cuda_fill_bytes(char *dst, const char *value_host, size_t elsize, long n) {
    if (n <= 0 || dst == NULL || value_host == NULL || elsize == 0) {
        return;
    }
    cudaMemcpy(dst, value_host, elsize, cudaMemcpyHostToDevice);
    long filled = 1;
    while (filled < n) {
        long copy_count = (filled * 2 <= n) ? filled : (n - filled);
        cudaMemcpy(dst + (size_t)filled * elsize,
                   dst,
                   (size_t)copy_count * elsize,
                   cudaMemcpyDeviceToDevice);
        filled += copy_count;
    }
}

void
vmemcheck() {
    if (MAIN_MEM_STACK.totalGPUAllocated != 0) {
        printf("\nVRAM MEMORY LEAK: leaked %d array(s)\n", MAIN_MEM_STACK.totalGPUAllocated);
    }
}

float
NDArray_VFLOAT(char *target) {
    float value;
    cudaMemcpy(&value, target, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

double NDArray_VDOUBLE(char *target) {
    double value;
    cudaMemcpy(&value, target, sizeof(double), cudaMemcpyDeviceToHost);
    return value;
}

float
NDArray_VFLOATF_I(float *target, int index) {
    float value;
    cudaMemcpy(&value, &(target[index]), sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

#endif
