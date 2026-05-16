#ifndef PHPSCI_NDARRAY_DEBUG_H
#define PHPSCI_NDARRAY_DEBUG_H

#include "ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif

void  NDArray_Dump(NDArray *array);
char *print_matrix_float64(double *buffer, int ndims, int *shape, int *strides, int num_elements, int device);
char *print_matrix_float32(float  *buffer, int ndims, int *shape, int *strides, int num_elements, int device);
char *print_matrix_generic(const char *type, const char *data,
                            int ndims, int *shape, int *strides,
                            long num_elements, int device);
void  NDArrayIterator_DUMP(NDArray *a);
void  NDArray_DumpDevices(void);

#ifdef __cplusplus
}
#endif

#endif /* PHPSCI_NDARRAY_DEBUG_H */
