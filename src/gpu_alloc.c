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
