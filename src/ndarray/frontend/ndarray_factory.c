// Z_TYPE_P,            IS_ARRAY,         IS_LONG,  IS_DOUBLE, IS_OBJECT,
// Z_OBJ_P,             Z_LVAL_P,         Z_DVAL_P, Z_OBJCE_P, ZSTR_VAL,
// instanceof_function, zend_throw_error
#include <Zend/zend_interfaces.h>

// ce, phpsci_ce_NDArray
#include "../../../php_numpower.h"

// NDARRAY_TYPE_FLOAT32, NDARRAY_TYPE_FLOAT64, …
#include "../../types.h"

// ndarray_set_from_double, ndarray_set_from_string
#include "../../ndarray_types.h"

// buffer_get
#include "../../buffer.h"

// NDArray
#include "ndarray_factory.h"

//PRIVATE

/**
 * @brief Get number of dimensions from php array
 *
 * @param ht A pointer to the zval array value
 * 
 * @return The number of dimensions in the array
 */
int _getNumDimsFromZval(zend_array *zendArray) {
    int num_dims = 0;

    if (zend_array_count(zendArray) == 0) {
        return 1;
    }

    zval *val = zend_hash_index_find(zendArray, 0);

    while (val && Z_TYPE_P(val) == IS_ARRAY) {
        num_dims++;
        val = zend_hash_index_find(Z_ARRVAL_P(val), 0);
    }

    return num_dims + 1;
}

/**
 * @brief Check if the zend array is packed
 *
 * @param arr A pointer to the zend array
 * 
 * @return 1 if the array is packed, 0 otherwise
 */
bool _isPackedZendArray(zend_array *zendArray) {
    return zendArray->nNumUsed == zendArray->nNumOfElements;
}

/**
 * @brief Count the shape of a zend array
 *
 * @param[in]    zendArray A pointer to the zend array
 * @param[inout] shape     A pointer to the shape array
 * @param[in]    ndim      The number of dimensions
 */
void _countZendArrayShape(zend_array *zendArray, int *shape, int ndim) {
    int i;

    if (shape == NULL && ndim != 0) {
        return;
    }

    if (zend_array_count(zendArray) == 0) {
        return;
    }

    // Initialize shape array to zeros
    for (i = 0; i < ndim; i++) {
        shape[i] = 0;
    }

    // Traverse the zend array to get the shape
    zval *val;
    ZEND_HASH_FOREACH_VAL(zendArray, val)
            {
                if (Z_TYPE_P(val) == IS_ARRAY) {
                    _countZendArrayShape(Z_ARRVAL_P(val), shape + 1, ndim - 1);
                    shape[0]++;
                } else {
                    shape[0]++;
                }
            }
    ZEND_HASH_FOREACH_END();
}

/**
 * @brief Create a buffer for the NDArray
 *
 * @param[inout] ndarray     A pointer to the NDArray
 * @param        numElements The number of elements in the array
 * @param        elsize      The size of each element in bytes
 */
void _createBuffer(NDArray *ndarray, int numElements, int elsize) {
    ndarray->data = emalloc(numElements * elsize);
}

/**
 * @brief Convert a zval to double safely
 *
 * Converts a PHP zval to a C double, handling numeric types,
 * booleans and strings containing valid float literals.
 *
 * @param[in] val Pointer to the zval to convert.
 *            The value may be a number, boolean, or string.
 *            Other types will cause a runtime error.
 *
 * @return The resulting double value.
 *
 * @throws Error If the zval is of an unsupported type or
 *         the string cannot be parsed as a float.
 */
static double zval_to_double_safe(zval *val) {
    ZVAL_DEREF(val);
    switch (Z_TYPE_P(val)) {
        case IS_LONG:
            return (double) Z_LVAL_P(val);
        case IS_DOUBLE:
            return Z_DVAL_P(val);
        case IS_TRUE:
            return 1.0;
        case IS_FALSE:
            return 0.0;
        case IS_STRING: {
            const char *str = Z_STRVAL_P(val);
            size_t len = Z_STRLEN_P(val);
            const char *end = NULL;

            double value = zend_strtod(str, &end);
            if (end != str + len) {
                zend_throw_error(NULL, "Cannot parse string '%s' as float", str);
                return 0.0;
            }
            return value;
        }
        default:
            zend_throw_error(NULL, "Invalid type in NDArray initialization");
            return 0.0;
    }
}

static void set_ndarray_value(NDArray *ndarray, int index, double value) {
    ndarray_set_from_double(NDArray_TYPE(ndarray), (char *)NDArray_DATA(ndarray), (size_t)index, value);
}

static void set_ndarray_value_string(NDArray *ndarray, int index, const char *str) {
    ndarray_set_from_string(NDArray_TYPE(ndarray), (char *)NDArray_DATA(ndarray), (size_t)index, str);
}

/**
 * @brief Fill the NDArray from a zend array (recursive)
 *
 * @param[in,out] ndarray     A pointer to the NDArray
 * @param[in]    zendArray   A pointer to the zend array
 * @param[in,out] firstIndex  A pointer to the first index
 */
void _fillFromZendArray(NDArray *ndarray, zend_array *zendArray, int *firstIndex) {
    zval       *element;
    const char *type = NDArray_TYPE(ndarray);

    ZEND_HASH_FOREACH_VAL(zendArray, element)
    {
        ZVAL_DEREF(element);
        if (Z_TYPE_P(element) == IS_ARRAY) {
            _fillFromZendArray(ndarray, Z_ARRVAL_P(element), firstIndex);
            continue;
        }

        /* String path: lossless for exotic / large-integer types */
        if (Z_TYPE_P(element) == IS_STRING && type_needs_string_io(type)) {
            set_ndarray_value_string(ndarray, *firstIndex, Z_STRVAL_P(element));
            ++(*firstIndex);
            continue;
        }

        /* Integer path: avoid double-rounding for large int64/uint64 values. */
        if (Z_TYPE_P(element) == IS_LONG &&
            (!strcmp(type, "int64") || !strcmp(type, "uint64") ||
             !strcmp(type, "int32") || !strcmp(type, "uint32") ||
             !strcmp(type, "int16") || !strcmp(type, "uint16") ||
             !strcmp(type, "int8")  || !strcmp(type, "uint8"))) {
            zend_long lv = Z_LVAL_P(element);
            if (!strcmp(type, "int64") || !strcmp(type, "uint64")) {
                char tmp[32];
                snprintf(tmp, sizeof(tmp), "%lld", (long long)lv);
                ndarray_set_from_string(type, (char *)NDArray_DATA(ndarray), (size_t)*firstIndex, tmp);
            } else {
                ndarray_set_from_double(type, (char *)NDArray_DATA(ndarray), (size_t)*firstIndex, (double)lv);
            }
            ++(*firstIndex);
            continue;
        }

        /* Default path: convert via double */
        double value = zval_to_double_safe(element);
        if (EG(exception)) return;
        set_ndarray_value(ndarray, *firstIndex, value);
        ++(*firstIndex);
    } ZEND_HASH_FOREACH_END();
}

/**
 * @brief Creates an NDArray object from a zval php array.
 *
 * This function takes a PHP array (zval) and creates an NDArray from it.
 * The type parameter is used to specify the desired data type of the NDArray.
 *
 * @param[in] ht    A pointer to the zval array value to be converted to an NDArray.
 * @param     ndim  A number of PHP array dimensions.
 * @param[in] type  A string representing the desired data type for the NDArray.
 * 
 * @return A pointer to the newly created NDArray, or NULL if the zval is not an array.
 */
NDArray *_createFromZendArray(zend_array *ht, const char *type) {
    int last_index = 0;
    int *shape;

    int ndim = _getNumDimsFromZval(ht);

    if (ndim != 0) {
        shape = ecalloc(ndim, sizeof(int));
    } else {
        shape = ecalloc(1, sizeof(int));
    }

    if (!_isPackedZendArray(ht)) {
        return NULL;
    }

    _countZendArrayShape(ht, shape, ndim);
    int total_num_elements = shape[0];

    // Calculate number of elements
    for (int i = 1; i < ndim; i++) {
        total_num_elements = total_num_elements * shape[i];
    }

    NDArray *array = NDArray_create(shape, ndim, type, NDARRAY_DEVICE_CPU);

    if (ndim != 0) {
        _createBuffer(array, total_num_elements, get_type_size(type));
        _fillFromZendArray(array, ht, &last_index);
    } else {
        array->data = NULL;
        array->descriptor->numElements = 0;
    }

    add_to_buffer(array);

    return array;
}

/**
 * @brief Creates a float32 NDArray from a long scalar.
 *
 * @param scalar The long scalar value to be converted to a float32 NDArray.
 * 
 * @return A pointer to the newly created NDArray.
 */
NDArray *_createFloat32FromLongScalar(long scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->uuid = -1;
    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(float);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT32;
    rtn->data = emalloc(sizeof(float));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((float *) rtn->data)[0] = (float) scalar;

    add_to_buffer(rtn);

    return rtn;
}

/**
 * @brief Creates a float32 NDArray from a double scalar.
 * 
 * @param scalar The double scalar value to be converted to a float32 NDArray.
 * 
 * @return A pointer to the newly created NDArray.
 */
NDArray *_createFloat32FromDoubleScalar(double scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->uuid = -1;
    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(float);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT32;
    rtn->data = emalloc(sizeof(float));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((float *) rtn->data)[0] = (float) scalar;

    add_to_buffer(rtn);

    return rtn;
}

/**
 * @brief Creates a double64 NDArray from a long scalar.
 * 
 * @param scalar The long scalar value to be converted to a double64 NDArray.
 * 
 * @return A pointer to the newly created NDArray.
 */
NDArray *_createDouble64FromLongScalar(long scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->uuid = -1;
    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(double);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT64;
    rtn->data = emalloc(sizeof(double));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((double *) rtn->data)[0] = (double) scalar;

    add_to_buffer(rtn);

    return rtn;
}

/**
 * @brief Creates a double64 NDArray from a double scalar.
 * 
 * @param scalar The double scalar value to be converted to a double64 NDArray.
 * 
 * @return A pointer to the newly created NDArray.
 */
NDArray *_createDouble64FromDoubleScalar(double scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->uuid = -1;
    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(double);
    rtn->descriptor->type = NDARRAY_TYPE_FLOAT64;
    rtn->data = emalloc(sizeof(double));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator = NULL;
    rtn->base = NULL;
    rtn->refcount = 1;
    ((double *) rtn->data)[0] = scalar;

    add_to_buffer(rtn);

    return rtn;
}

// PUBLIC

/**
 * @brief Get the UUID of an object.
 * 
 * @param obj A pointer to the zval object.
 * 
 * @return The UUID of the object.
 */
int getObjectUuid(zval *obj) {
    return Z_LVAL_P(OBJ_PROP_NUM(Z_OBJ_P(obj), 0));
}

/**
 * @brief Creates an NDArray object from a zval php array.
 * 
 * @param[in] obj  A pointer to the zval object to be converted to an NDArray.
 * @param[in] type A string representing the desired data type for the NDArray.
 * 
 * @return A pointer to the newly created NDArray, or NULL if the zval is not an array.
 */
/* Create a 0-dim (scalar) NDArray of [type] from a double value */
static NDArray *_createScalarFromDouble(double val, const char *type) {
    int elsize = get_type_size(type);
    if (elsize == 0) return NULL;

    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);
    rtn->uuid       = -1;
    rtn->ndim       = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize      = elsize;
    rtn->descriptor->type        = type;
    rtn->data       = emalloc((size_t)elsize);
    rtn->device     = NDARRAY_DEVICE_CPU;
    rtn->strides    = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator   = NULL;
    rtn->base       = NULL;
    rtn->refcount   = 1;
    ndarray_set_from_double(type, rtn->data, 0, val);
    add_to_buffer(rtn);
    return rtn;
}

/* Create a 0-dim (scalar) NDArray of [type] from a string (lossless for exotic types) */
static NDArray *_createScalarFromString(const char *str, const char *type) {
    int elsize = get_type_size(type);
    if (elsize == 0) return NULL;

    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);
    rtn->uuid       = -1;
    rtn->ndim       = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize      = elsize;
    rtn->descriptor->type        = type;
    rtn->data       = emalloc((size_t)elsize);
    rtn->device     = NDARRAY_DEVICE_CPU;
    rtn->strides    = emalloc(sizeof(int));
    rtn->dimensions = emalloc(sizeof(int));
    rtn->iterator   = NULL;
    rtn->base       = NULL;
    rtn->refcount   = 1;
    ndarray_set_from_string(type, rtn->data, 0, str);
    add_to_buffer(rtn);
    return rtn;
}

NDArray *NDArrayFactory_createFromZval(zval *obj, const char *type) {
    if (Z_TYPE_P(obj) == IS_ARRAY) {
        return _createFromZendArray(Z_ARRVAL_P(obj), type);
    }

    /* Legacy fast paths kept for float32 / float64 */
    if (Z_TYPE_P(obj) == IS_LONG && is_type(type, NDARRAY_TYPE_FLOAT32)) {
        return _createFloat32FromLongScalar(Z_LVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_DOUBLE && is_type(type, NDARRAY_TYPE_FLOAT32)) {
        return _createFloat32FromDoubleScalar(Z_DVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_LONG && is_type(type, NDARRAY_TYPE_FLOAT64)) {
        return _createDouble64FromLongScalar(Z_LVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_DOUBLE && is_type(type, NDARRAY_TYPE_FLOAT64)) {
        return _createDouble64FromDoubleScalar(Z_DVAL_P(obj));
    }

    /* Generic scalar path for all other types */
    if (Z_TYPE_P(obj) == IS_LONG) {
        return _createScalarFromDouble((double)Z_LVAL_P(obj), type);
    }
    if (Z_TYPE_P(obj) == IS_DOUBLE) {
        return _createScalarFromDouble(Z_DVAL_P(obj), type);
    }
    if (Z_TYPE_P(obj) == IS_STRING) {
        if (type_needs_string_io(type)) {
            return _createScalarFromString(Z_STRVAL_P(obj), type);
        }
        return _createScalarFromDouble(zval_to_double_safe(obj), type);
    }

    if (Z_TYPE_P(obj) == IS_OBJECT) {
        zend_class_entry *ce = Z_OBJCE_P(obj);
        if (instanceof_function(ce, phpsci_ce_NDArray)) {
            return buffer_get(getObjectUuid(obj));
        }
#ifdef HAVE_GD
        zend_string *class_name = Z_OBJ_P(obj)->ce->name;
        if (strcmp(ZSTR_VAL(class_name), "GdImage") == 0) {
            NDArray *ndarray = NDArray_FromGD(obj, false);
            add_to_buffer(ndarray);
            return ndarray;
        }
#endif
    }

    zend_throw_error(NULL, "Argument must be an array of numerics, float, int, string, GdImage or NDArray.");
    return NULL;
}

/**
 * @brief Restores an NDArray from a zval object.
 * 
 * @param zvalNdarray A pointer to the zval object to be restored.
 * 
 * @return A pointer to the restored NDArray, or NULL if the zval is not an NDArray.
 */
NDArray *NDArrayFactory_restoreFromZval(zval *zvalNdarray) {
    if (Z_TYPE_P(zvalNdarray) == IS_OBJECT) {
        zend_class_entry *ce = Z_OBJCE_P(zvalNdarray);
        if (instanceof_function(ce, phpsci_ce_NDArray)) {
            return buffer_get(getObjectUuid(zvalNdarray));
        }
    }

    zend_throw_error(NULL, "Argument must be an NDArray.");
    return NULL;
}