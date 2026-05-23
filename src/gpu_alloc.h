#ifndef NUMPOWER_GPU_ALLOC_H
#define NUMPOWER_GPU_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

void vmalloc(void **target, unsigned int size);
void vfree(void *target);
void vmemcheck();
void vmemcpyd2d(char* target, char* dst, unsigned int size);
void vmemcpyh2d(char* target, char* dst, unsigned int size);

/**
 * @brief Broadcast a single value across a VRAM buffer with no host
 *        intermediate beyond the @p value_host source.
 *
 * Used by typed initializers (`NDArray_Ones`, future `NDArray_FullTyped`)
 * to fill an NDArray's on-device storage with the dtype-appropriate
 * encoded scalar. The work is done entirely on the device via a
 * doubling-prefix `cudaMemcpyDeviceToDevice` loop — host RAM never holds
 * more than the @p value_host source provided by the caller (≤ 16 bytes
 * for the widest supported dtype, fp128 in DD form).
 *
 * The loop performs O(log n) device-to-device copies: pass k doubles the
 * length of the already-filled prefix until the buffer is full. Each
 * `cudaMemcpyDeviceToDevice` call has constant launch overhead but
 * benefits from the device's full memory bandwidth, so the broadcast is
 * bandwidth-bound for any non-trivial @p n.
 *
 * @param[in,out] dst        Device pointer; first @p n elements are written.
 * @param[in]     value_host Host pointer to one element of @p elsize bytes.
 * @param[in]     elsize     Bytes per element (must be > 0).
 * @param[in]     n          Number of elements to write. n <= 0 is a no-op.
 */
void cuda_fill_bytes(char *dst, const char *value_host, size_t elsize, long n);

float NDArray_VFLOAT(char *target);
float NDArray_VFLOATF_I(float *target, int index);

double NDArray_VDOUBLE(char *target);

#ifdef __cplusplus
}
#endif

#endif //NUMPOWER_GPU_ALLOC_H
