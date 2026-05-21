#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "debug.h"
#ifndef _MSC_VER
#include "../config.h"
#endif
#include "ndarray.h"
#include "ndarray_types.h"
#include "types.h"
#include <string.h>

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
/* nvml.h was historically pulled in here but no NVML symbols are referenced
   anywhere in the codebase. Dropping it avoids needing the NVIDIA Management
   Library dev pack (nvml_dev) on Windows CI just to satisfy a dead include. */
#endif

/**
 * Dump NDArray
 */
void
NDArray_Dump(NDArray* array) {
    int i;
    php_printf("\n=================================================");
    php_printf("\nNDArray.uuid\t\t\t%d", array->uuid);
    php_printf("\nNDArray.ndim\t\t\t%d", array->ndim);
    php_printf("\nNDArray.dims\t\t\t[");
    for(i = 0; i < array->ndim; i ++) {
        php_printf(" %d", array->dimensions[i]);
    }
    php_printf(" ]\n");
    php_printf("NDArray.strides\t\t\t[");
    for(i = 0; i < array->ndim; i ++) {
        php_printf(" %d", array->strides[i]);
    }
    php_printf(" ]\n");
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        php_printf("NDArray.device\t\t\t(%d) %s\n", NDArray_DEVICE(array), "GPU");
    } else if(NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
        php_printf("NDArray.device\t\t\t(%d) %s\n", NDArray_DEVICE(array), "CPU");
    } else {
        php_printf("NDArray.device\t\t\t(%d) %s\n", NDArray_DEVICE(array), "ERROR");
    }
    php_printf("NDArray.refcount\t\t%d\n", array->refcount);
    php_printf("NDArray.descriptor.elsize\t%d\n", array->descriptor->elsize);
    php_printf("NDArray.descriptor.numElements\t%ld\n", array->descriptor->numElements);
    php_printf("NDArray.descriptor.type\t\t%s\n", array->descriptor->type);
    php_printf("NDArray.iterator.current_index\t%d", array->iterator->currentIndex);
    php_printf("\n=================================================\n");
}

/**
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 * @param cur_dim
 * @param index
 * @return
 */
char*
print_array_float32(float* buffer, int ndims, int* shape, int* strides, int cur_dim, int* index, int num_elements, int* padded) {
    char* str;
    int i, j, t;
    int reverse_run = 0;

    if (num_elements == 0) {
        str = (char*)emalloc(3 * sizeof(char));
        str = "[]";
        return str;
    }

    // Allocate memory for the string
    str = (char*)emalloc(10000000 * sizeof(char));
    if (str == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for string.\n");
        exit(1);
    }

    if (ndims == 0) {
        sprintf(str, "%g\n", buffer[0]);
        return str;
    }

    if (cur_dim == ndims - 1) {
        // Print the opening bracket for this dimension
        sprintf(str, "[");
        // Print the elements of the array
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            // Compute the offset of this element in the buffer
            int offset = 0;
            for (int k = 0; k < ndims; k++) {
                offset += index[k] * strides[k];
            }
            // Print the element
            sprintf(str + strlen(str), "%.8g", buffer[offset / sizeof(float)]);

            // Print a comma if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                sprintf(str + strlen(str), ", ");
            }

            if ((i + 1) % 10 == 0 && i < shape[cur_dim] - 1) {
                sprintf(str + strlen(str), "\n");
                for (t = 0; t < ndims; t++) {
                    sprintf(str + strlen(str), " ");
                }
            }

            if (shape[cur_dim] > 20) {
                if (i > 1 && reverse_run == 0) {
                    i = shape[cur_dim] - 4;
                    reverse_run = 1;
                    sprintf(str + strlen(str), "... ");
                }
            }
        }

        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");
        if (cur_dim > 0 && index[cur_dim-1] < shape[ndims - 2] - 1) {
            sprintf(str + strlen(str), "\n ");
        }
    } else {
        if (cur_dim != 0) {
            if (cur_dim == index[cur_dim - 1]) {
                for (t = cur_dim; t < ndims; t++) {
                    sprintf(str + strlen(str), " ");
                }
            }
        }
        // Print the opening bracket for this dimension
        sprintf(str, "[");

        // Recursively print each element in the dimension
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            char* child_str = print_array_float32(buffer, ndims, shape, strides, cur_dim + 1, index, num_elements,
                                                  padded);

            // Add the child string to the parent string
            sprintf(str + strlen(str), "%s", child_str);

            // Free the child string
            efree(child_str);

            // Print a comma and newline if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                for (j = 0; j < cur_dim; j++) {
                    sprintf(str + strlen(str), " ");
                }
            }

            if (ndims > 1) {
                if (shape[ndims - 1] * shape[ndims - 2] > 500 && shape[cur_dim] > 10) {
                    if(i >= 2 && reverse_run == 0) {
                        i = shape[cur_dim] - 4;
                        reverse_run = 1;
                        sprintf(str + strlen(str), "...\n");
                        if (i < shape[cur_dim] - 1) {
                            for (j = 1; j < ndims; j++) {
                                sprintf(str + strlen(str), " ");
                            }
                        }
                        *padded = 1;
                    }
                }
            }
        }
        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");

        if (cur_dim != 0 && index[cur_dim-1] < shape[cur_dim-1] - 1) {
            sprintf(str + strlen(str), "\n");
        }
    }

    return str;
}


/**
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 * @param cur_dim
 * @param index
 * @return
 */
char*
print_array_float64(double* buffer, int ndims, int* shape, int* strides, int cur_dim, int* index, int num_elements, int* padded) {
    char* str;
    int i, j, t;
    int reverse_run = 0;

    if (num_elements == 0) {
        str = "[]";
        return str;
    }

    // Allocate memory for the string
    str = (char*)emalloc(num_elements * 64 * sizeof(char));
    if (str == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for string.\n");
        exit(1);
    }

    if (ndims == 0) {
        sprintf(str, "%g\n", buffer[0]);
        return str;
    }

    if (cur_dim == ndims - 1) {
        // Print the opening bracket for this dimension
        sprintf(str, "[");
        // Print the elements of the array
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            // Compute the offset of this element in the buffer
            int offset = 0;
            for (int k = 0; k < ndims; k++) {
                offset += index[k] * strides[k];
            }
            // Print the element
            sprintf(str + strlen(str), "%.16g", buffer[offset / sizeof(double)]);

            // Print a comma if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                sprintf(str + strlen(str), ", ");
            }

            if ((i + 1) % 10 == 0 && i < shape[cur_dim] - 1) {
                sprintf(str + strlen(str), "\n");
                for (t = 0; t < ndims; t++) {
                    sprintf(str + strlen(str), " ");
                }
            }

            if (shape[cur_dim] > 20) {
                if (i > 1 && reverse_run == 0) {
                    i = shape[cur_dim] - 4;
                    reverse_run = 1;
                    sprintf(str + strlen(str), "... ");
                }
            }
        }

        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");
        if (cur_dim > 0 && index[cur_dim-1] < shape[ndims - 2] - 1) {
            sprintf(str + strlen(str), "\n ");
        }
    } else {
        if (cur_dim != 0) {
            if (cur_dim == index[cur_dim - 1]) {
                for (t = cur_dim; t < ndims; t++) {
                    sprintf(str + strlen(str), " ");
                }
            }
        }
        // Print the opening bracket for this dimension
        sprintf(str, "[");

        // Recursively print each element in the dimension
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            char* child_str = print_array_float64(buffer, ndims, shape, strides, cur_dim + 1, index, num_elements,
                                                  padded);

            // Add the child string to the parent string
            sprintf(str + strlen(str), "%s", child_str);

            // Free the child string
            efree(child_str);

            // Print a comma and newline if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                for (j = 0; j < cur_dim; j++) {
                    sprintf(str + strlen(str), " ");
                }
            }

            if (ndims > 1) {
                if (shape[ndims - 1] * shape[ndims - 2] > 500 && shape[cur_dim] > 10) {
                    if(i >= 2 && reverse_run == 0) {
                        i = shape[cur_dim] - 4;
                        reverse_run = 1;
                        sprintf(str + strlen(str), "...\n");
                        if (i < shape[cur_dim] - 1) {
                            for (j = 1; j < ndims; j++) {
                                sprintf(str + strlen(str), " ");
                            }
                        }
                        *padded = 1;
                    }
                }
            }
        }
        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");

        if (cur_dim != 0 && index[cur_dim-1] < shape[cur_dim-1] - 1) {
            sprintf(str + strlen(str), "\n");
        }
    }

    return str;
}

/**
 * Print matrix of type float32
 *
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 */
char*
print_matrix_float32(float* buffer, int ndims, int* shape, int* strides, int num_elements, int device) {
    float *tmp_buffer;
    int *index = emalloc(ndims * sizeof(int));
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        tmp_buffer = emalloc(num_elements * sizeof(float));
        cudaMemcpy(tmp_buffer, buffer, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    } else {
        tmp_buffer = buffer;
    }
    int padded = 0;
    char* rtn = print_array_float32(tmp_buffer, ndims, shape, strides, 0, index, num_elements, &padded);
    efree(index);
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        efree(tmp_buffer);
    }
#endif
    return rtn;
}

char*
print_matrix_float64(double* buffer, int ndims, int* shape, int* strides, int num_elements, int device) {
    double *tmp_buffer;
    int *index = emalloc(ndims * sizeof(int));
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        tmp_buffer = emalloc(num_elements * sizeof(double));
        cudaMemcpy(tmp_buffer, buffer, num_elements * sizeof(double), cudaMemcpyDeviceToHost);
#endif
    } else {
        tmp_buffer = buffer;
    }
    int padded = 0;
    char* rtn = print_array_float64(tmp_buffer, ndims, shape, strides, 0, index, num_elements, &padded);
    efree(index);
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        efree(tmp_buffer);
    }
#endif
    return rtn;
}

/* ── Generic array printer for all non-float32/float64 dtypes ─────────────
   Uses ndarray_element_to_string() so it handles every supported type.   */

static char *print_array_generic(
    const char *type,
    const char *data,
    int ndims, int *shape, int *strides,
    int cur_dim, int *index,
    long num_elements, int *padded)
{
    char elem_buf[64];
    int  i, j, t;
    int  reverse_run = 0;

    if (num_elements == 0) {
        char *s = (char *)emalloc(3);
        strcpy(s, "[]");
        return s;
    }

    /* Allocate a buffer large enough for this sub-array.
       Each element needs at most 48 chars + separators. */
    size_t max_sz = (size_t)num_elements * 52 + (size_t)ndims * 8 + 64;
    if (max_sz < 256) max_sz = 256;
    char *str = (char *)emalloc(max_sz);
    if (!str) return NULL;
    str[0] = '\0';

    if (ndims == 0) {
        ndarray_element_to_string(type, data, 0, elem_buf, sizeof(elem_buf));
        snprintf(str, max_sz, "%s\n", elem_buf);
        return str;
    }

    if (cur_dim == ndims - 1) {
        strcat(str, "[");
        for (i = 0; i < shape[cur_dim]; i++) {
            index[cur_dim] = i;
            int offset = 0;
            for (int k = 0; k < ndims; k++) offset += index[k] * strides[k];
            ndarray_element_to_string(type, data, (size_t)offset, elem_buf, sizeof(elem_buf));
            strcat(str, elem_buf);
            if (i < shape[cur_dim] - 1) strcat(str, ", ");
            if ((i + 1) % 10 == 0 && i < shape[cur_dim] - 1) {
                strcat(str, "\n");
                for (t = 0; t < ndims; t++) strcat(str, " ");
            }
            if (shape[cur_dim] > 20 && i > 1 && reverse_run == 0) {
                i = shape[cur_dim] - 4;
                reverse_run = 1;
                strcat(str, "... ");
            }
        }
        strcat(str, "]");
        if (cur_dim > 0 && index[cur_dim - 1] < shape[ndims - 2] - 1) strcat(str, "\n ");
    } else {
        strcat(str, "[");
        for (i = 0; i < shape[cur_dim]; i++) {
            index[cur_dim] = i;
            char *child = print_array_generic(type, data, ndims, shape, strides,
                                              cur_dim + 1, index, num_elements, padded);
            strcat(str, child);
            efree(child);
            if (i < shape[cur_dim] - 1) {
                for (j = 0; j < cur_dim; j++) strcat(str, " ");
            }
            if (ndims > 1 && shape[ndims - 1] * shape[ndims - 2] > 500 && shape[cur_dim] > 10) {
                if (i >= 2 && reverse_run == 0) {
                    i = shape[cur_dim] - 4;
                    reverse_run = 1;
                    strcat(str, "...\n");
                    if (i < shape[cur_dim] - 1) {
                        for (j = 1; j < ndims; j++) strcat(str, " ");
                    }
                    *padded = 1;
                }
            }
        }
        strcat(str, "]");
        if (cur_dim != 0 && index[cur_dim - 1] < shape[cur_dim - 1] - 1) strcat(str, "\n");
    }

    return str;
}

char *print_matrix_generic(
    const char *type,
    const char *data,
    int ndims, int *shape, int *strides,
    long num_elements, int device)
{
    const char *tmp = data;
    char       *gpu_buf = NULL;
    int         elsize = get_type_size(type);

    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        gpu_buf = (char *)emalloc((size_t)num_elements * (size_t)elsize);
        /* fp128 lives on GPU as double-double (hi, lo) — go through the typed
           D2H helper so it's reassembled into native __float128 bytes for the
           generic stringifier. Other dtypes use the same byte layout on both
           devices, so the helper degenerates to a plain cudaMemcpy. */
        NDArray_TypedD2H(gpu_buf, data, num_elements, type);
        tmp = gpu_buf;
#endif
    }

    int *index = (int *)emalloc((size_t)(ndims > 0 ? ndims : 1) * sizeof(int));
    memset(index, 0, (size_t)(ndims > 0 ? ndims : 1) * sizeof(int));
    int  padded = 0;
    char *rtn = print_array_generic(type, tmp, ndims, shape, strides, 0, index, num_elements, &padded);
    efree(index);

#ifdef HAVE_CUBLAS
    if (gpu_buf) efree(gpu_buf);
#endif
    return rtn;
}

void
NDArray_DumpDevices() {
#ifdef HAVE_CUBLAS
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("Failed to retrieve device count: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("\n==============================================================================\n");
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        struct cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);

        if (err != cudaSuccess) {
            printf("Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            return;
        }
        printf("\n---------------------------------------------------------------------------");
        printf("\nDevice %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Multi processor count: %d\n", deviceProp.multiProcessorCount);
        printf("  Max threads in X-dimension of block: %d\n", deviceProp.maxThreadsDim[0]);
        printf("  Max threads in Y-dimension of block: %d\n", deviceProp.maxThreadsDim[1]);
        printf("  Max threads in Z-dimension of block: %d\n", deviceProp.maxThreadsDim[2]);
        printf("  Max grid size in X-dimension: %d\n", deviceProp.maxGridSize[0]);
        printf("  Max grid size in Y-dimension: %d\n", deviceProp.maxGridSize[1]);
        printf("  Max grid size in Z-dimension: %d\n", deviceProp.maxGridSize[2]);
        printf("---------------------------------------------------------------------------\n");
    }
    printf("\n==============================================================================\n");
#else
    php_printf("\nNo GPU devices available. CUDA not enabled.\n");
#endif
}

/**
 * @param a
 */
void
NDArrayIterator_DUMP(NDArray *a) {
    printf("\n====================================\n");
    printf("iterator.current_index:\t\t%d",a->iterator->currentIndex);
    printf("\n====================================\n");
}