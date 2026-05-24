#ifndef PHPSCI_NDARRAY_NDARRAY_H
#define PHPSCI_NDARRAY_NDARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"
#include <Zend/zend_types.h>
#include <stdbool.h>
#include "ndarray_types.h"

#define NDARRAY_MAX_DIMS 128
#define NDARRAY_ARRAY_C_CONTIGUOUS    0x0001
#define NDARRAY_ARRAY_F_CONTIGUOUS    0x0002

#define NDArray_UUID(a) ((int)((a)->uuid))
#define NDArray_DATA(a) ((void *)((a)->data))
#define NDArray_DESCRIPTOR(a) ((NDArrayDescriptor *)((a)->descriptor))
#define NDArray_F64DATA(a) ((double *)((a)->data))
#define NDArray_F32DATA(a) ((float *)((a)->data))
#define NDArray_F128DATA(a) ((ndarray_fp128_t *)((a)->data))
#define NDArray_NDIM(a) ((int)((a)->ndim))
#define NDArray_FLAGS(a) ((int)((a)->flags))
#define NDArray_SHAPE(a) ((int *)((a)->dimensions))
#define NDArray_STRIDES(a) ((int *)((a)->strides))
#define NDArray_TYPE(a) ((const char *)((a)->descriptor->type))
#define NDArray_NUMELEMENTS(a) ((long)((a)->descriptor->numElements))
#define NDArray_ELSIZE(a) ((int)((a)->descriptor->elsize))
#define NDArray_DEVICE(a) ((int)((a)->device))

#define NDArray_ADDREF(a) ((a)->refcount++)
#define NDArray_DELREF(a) ((a)->refcount--)

#define NDARRAY_DEVICE_CPU 0
#define NDARRAY_DEVICE_GPU 1

/**
 * NDArray Dims
 * 
 * @property ptr Pointer to the dimensions array
 * @property len Length of the dimensions array
 **/
typedef struct NDArray_Dims {
    int* ptr;
    int  len;
} NDArray_Dims;

/**
 * NDArray Iterator
 * 
 * @property currentIndex Current index of the iterator
 */
typedef struct NDArrayIterator {
    int currentIndex;
} NDArrayIterator;

/**
 * NDArray Descriptor
 * 
 * @property type        NDArray datatype
 * @property elsize      NDArray datatype size
 * @property numElements Number of elements in the NDArray
 */
typedef struct NDArrayDescriptor {
    const char* type;
    int         elsize;
    long        numElements; 
} NDArrayDescriptor;

/**
 * NDArray Stride Sort Item
 * 
 * @property perm   Permutation index
 * @property stride Stride value
 */
typedef struct {
    int perm, stride;
} NDArrayStrideSortItem;

/**
 * NDArray
 * 
 * @property uuid         Unique identifier for the NDArray
 * @property strides      Strides vector (number of bytes)
 * @property dimensions   Dimensions size vector (Shape)
 * @property ndim         Number of Dimensions
 * @property data         Data Buffer (contiguous strided)
 * @property base         Used when sharing memory from other NDArray (slices, etc)
 * @property flags        Describes NDArray memory approach (Memory related flags)
 * @property descriptor   NDArray data descriptor
 * @property iterator     Iterator for the NDArray
 * @property php_iterator PHP iterator for the NDArray
 * @property refcount     Reference count for the NDArray
 * @property device       NDArray Device (0 = CPU, 1 = GPU)
 */
typedef struct NDArray {
    int                uuid;
    int*               strides;
    int*               dimensions;
    int                ndim;
    char*              data;
    struct NDArray*    base;
    int                flags;
    NDArrayDescriptor* descriptor;
    NDArrayIterator*   iterator;
    NDArrayIterator*   php_iterator;
    int                refcount;
    int                device;
} NDArray;

NDArray* NDArray_create(int *shape, int ndim, const char *type, int device);

/**
 * Enable flags for NDArray
 * 
 * @param[inout] arr Pointer to the NDArray
 * @param        flags Flags to enable
 */
void NDArray_enableFlags(NDArray * arr, int flags);

/**
 * Check if the NDArray has the specified flags
 * 
 * @param[in] arr   Pointer to the NDArray
 * @param     flags Flags to check
 * 
 * @return trye if the flags are set, false otherwise
 */
bool NDArray_checkFlags(const NDArray *arr, int flags);

/**
 * Clears the specified array flags. Does no checking,
 * assumes you know what you're doing.
 * 
 * @param[inout] arr Pointer to the NDArray
 * @param        flags Flags to clear
 */
void NDArray_CLEARFLAGS(NDArray *arr, int flags);

/**
 * Remove NDArary from the memory
 * 
 * @param[inout] array Pointer to the NDArray
 */
void NDArray_FREE(NDArray *array);

/**
 * Print NDArray information data
 * 
 * @param[in] array     Pointer to the NDArray
 * @param     do_return Flag to return the string or print it
 */
char* NDArray_Print(NDArray *array, int do_return);
NDArray *reduce(NDArray *array, int *axis, NDArray *(*operation)(NDArray *, NDArray *));
NDArray *single_reduce(NDArray *array, int *axis, float (*operation)(NDArray *));
float NDArray_Min(NDArray *target);
float NDArray_Max(NDArray *target);
NDArray* NDArray_Maximum(NDArray *a, NDArray *b);
NDArray * NDArray_Minimum(NDArray *a, NDArray *b);
NDArray* NDArray_MaxAxis(NDArray* target, int axis);
zval NDArray_ToPHPArray(NDArray *target);
/* Convert an NDArray scalar (0-dim) into a PHP zval of the appropriate native
   type for the dtype: IS_STRING for float128/uint64, IS_LONG for int/uint
   (other than uint64), IS_DOUBLE for float4..float64. Handles GPU memory. */
void NDArray_ScalarToZval(NDArray *array, zval *return_value);
int *NDArray_ToIntVector(NDArray *nda);
NDArray *NDArray_ToGPU(NDArray *target);
NDArray *NDArray_ToCPU(NDArray *target);
int NDArray_ShapeCompare(NDArray *a, NDArray *b);
NDArray* NDArray_Broadcast(NDArray *a, NDArray *b);
int NDArray_IsBroadcastable(const NDArray *arr1, const NDArray *arr2);
float NDArray_GetFloatScalar(NDArray *a);
double NDArray_GetDoubleScalar(NDArray *a);
void NDArray_FREEDATA(NDArray *target);
int NDArray_Overwrite(NDArray *target, NDArray *values);
NDArray* NDArray_FromGD(zval *a, bool channel_last, const char *dtype, int device);
void NDArray_ToGD(NDArray *a, NDArray *n_alpha, zval *output);
void NDArray_Save(NDArray *a, char * filename, int length);
NDArray* NDArray_Load(char * filename);
NDArray* NDArray_AssignRawScalar(NDArray *dst, NDArray *src);
int NDArray_AssignArray(NDArray *dst, NDArray *src);
int NDArray_CompareLists(int const *l1, int const *l2, int n);
void NDArray_CreateMultiSortedStridePerm(int narrays, NDArray **arrays, int ndim, int *out_strideperm);
void NDArray_CreateSortedStridePerm(int ndim, int const *strides, NDArrayStrideSortItem *out_strideperm);
NDArray *NDArray_AsType(NDArray *src, const char *target_type);
/* Copy n elements of typed data from CPU staging to GPU target. Handles the
   fp128 ↔ dd conversion transparently so callers don't need to know about
   the on-GPU representation. */
void NDArray_TypedH2D(char *gpu_dst, const char *cpu_src, long n, const char *dtype);
void NDArray_TypedD2H(char *cpu_dst, const char *gpu_src, long n, const char *dtype);

#ifdef __cplusplus
}
#endif

typedef float (*ElementWiseDoubleOperation)(float);
typedef float (*ElementWiseFloatOperation2F)(float, float, float);
typedef float (*ElementWiseFloatOperation1F)(float, float);

typedef double (*ElementWiseRealDoubleOperation)(double);

NDArray* NDArray_Map(NDArray *array, ElementWiseDoubleOperation op);
NDArray* NDArray_Map_Zval(NDArray *array, zval *callback);
NDArray* NDArray_Map2F(NDArray *array, ElementWiseFloatOperation2F op, float val1, float val2);
NDArray* NDArray_Map1F(NDArray *array, ElementWiseFloatOperation1F op, float val1);
NDArray* NDArray_Map1ND(NDArray *array, ElementWiseFloatOperation1F op, NDArray *val1);

NDArray* NDArray_Map_Double(NDArray *array, ElementWiseRealDoubleOperation op);

#endif //PHPSCI_NDARRAY_NDARRAY_H
