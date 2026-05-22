#ifndef PHPSCI_NDARRAY_BUFFER_H
#define PHPSCI_NDARRAY_BUFFER_H
#include "ndarray.h"

/**
 * MemoryStack : The memory buffer of CArrays
 */
struct MemoryStack {
    NDArray** buffer;   // Dynamic array to store NDArray pointers
    int bufferSize;     // Current size of the buffer
    int numElements;
    int *freeList;      // Stack of freed UUIDs available for reuse
    int freeListSize;   // Capacity of freeList
    int freeListTop;    // Index of next slot to pop (-1 = empty)
    int totalGPUAllocated;
    int totalAllocated;
    int totalFreed;
#ifdef ZTS
    MUTEX_T lock;
#endif
};

extern struct MemoryStack MAIN_MEM_STACK;

void buffer_ndarray_free(int uuid);
void add_to_buffer(NDArray* array);
void buffer_init(int size);
NDArray* buffer_get(int uuid);
void buffer_free();

/**
 * Atomically replace the NDArray at `uuid` and return the previous occupant.
 *
 * The caller is responsible for releasing the returned pointer via
 * NDArray_FREE — buffer_replace does NOT touch refcounts. On ZTS builds the
 * swap is performed under MAIN_MEM_STACK.lock so concurrent buffer_get/
 * buffer_ndarray_free calls never observe a torn state.
 *
 * @param uuid  Buffer slot to replace; must be in [0, numElements).
 * @param next  Replacement NDArray. Its `uuid` field is updated to `uuid`.
 * @return      Previous NDArray pointer, or NULL if `uuid` is out of range.
 */
NDArray* buffer_replace(int uuid, NDArray *next);
#endif //PHPSCI_NDARRAY_BUFFER_H
