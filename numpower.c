#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <Zend/zend_modules.h>
#include <Zend/zend_interfaces.h>
#include "php.h"
#include "ext/standard/info.h"
#include "numpower_arginfo.h"
#include "src/initializers.h"
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "src/buffer.h"
#include "src/iterators.h"
#include "php_numpower.h"
#include "src/debug.h"
#include "src/ndmath/arithmetics.h"
#include "src/logic.h"
#include "src/manipulation.h"
#include "src/ndmath/double_math.h"
#include "src/ndmath/linalg.h"
#include "config.h"
#include "src/types.h"
#include "src/indexing.h"
#include "src/ndmath/statistics.h"
#include "src/ndmath/signal.h"
#include "src/ndmath/calculation.h"
#include "src/dnn.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "src/ndmath/cuda/cuda_math.h"
#include "src/gpu_alloc.h"
#endif

#ifdef HAVE_GD

#endif

/* For compatibility with older PHP versions */
#ifndef ZEND_PARSE_PARAMETERS_NONE
#define ZEND_PARSE_PARAMETERS_NONE() \
	ZEND_PARSE_PARAMETERS_START(0, 0) \
	ZEND_PARSE_PARAMETERS_END()
#endif

static zend_object_handlers ndarray_object_handlers;
static zend_object_handlers numpower_object_handlers;
static zend_object_handlers arithmetic_object_handlers;

int *
zval_axis_argument(zval *arg, char *name, int *outsize)
{
    int *result;
    zend_string *key;
    zend_ulong idx;
    zval *val;
    *outsize = 0;
    if (Z_TYPE_P(arg) != IS_LONG && Z_TYPE_P(arg) != IS_ARRAY)
    {
        zend_throw_error(NULL, "`%s` argument must be either integer or an array of integers.", name);
        return NULL;
    }
    if (Z_TYPE_P(arg) == IS_LONG) {
        result = emalloc(sizeof(int));
        *result = zval_get_long(arg);
        *outsize = 1;
        return result;
    }

    result = emalloc(sizeof(int) * zend_array_count(Z_ARRVAL_P(arg)));
    ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(arg), idx, key, val) {
        if (Z_TYPE_P(val) != IS_LONG) {
            efree(result);
            zend_throw_error(NULL, "`%s` argument must be either integer or an array of integers.", name);
            return NULL;
        }
        result[idx] = zval_get_long(val);
        (*outsize)++;
    } ZEND_HASH_FOREACH_END();
    return result;
}

int
get_object_uuid(zval* obj) {
    return Z_LVAL_P(OBJ_PROP_NUM(Z_OBJ_P(obj), 0));
}

NDArray* ZVAL_TO_NDARRAY(zval* obj) {
    if (Z_TYPE_P(obj) == IS_ARRAY) {
        return Create_NDArray_FromZval(obj);
    }
    if (Z_TYPE_P(obj) == IS_LONG) {
        return NDArray_CreateFromLongScalar(Z_LVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_DOUBLE) {
        return NDArray_CreateFromDoubleScalar(Z_DVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_OBJECT) {
        zend_class_entry* ce = NULL;
        ce = Z_OBJCE_P(obj);
        zend_string* class_name = Z_OBJ_P(obj)->ce->name;
        if (strcmp(ZSTR_VAL(class_name), "NDArray") == 0) {
            if (ce == phpsci_ce_NDArray) {
                return buffer_get(get_object_uuid(obj));
            }
        }
#ifdef HAVE_GD
        /* Check if the zend_object class name is "GdImage" */
        if (strcmp(ZSTR_VAL(class_name), "GdImage") == 0) {
            return NDArray_FromGD(obj, false);
        }
#endif
    }
    zend_throw_error(NULL, "argument must be an array, long, double, gdimage or NDArray.");
    return NULL;
}

void CHECK_INPUT_AND_FREE(zval *a, NDArray *nda) {
    if (nda == NULL || a == NULL) {
        return;
    }
    if (Z_TYPE_P(a) == IS_ARRAY || Z_TYPE_P(a) == IS_DOUBLE || Z_TYPE_P(a) == IS_LONG) {
        NDArray_FREE(nda);
    }
#ifdef HAVE_GD
    if (Z_TYPE_P(a) == IS_OBJECT) {
        /* Check if the zend_object class name is "GdImage" */
        zend_string* class_name = Z_OBJ_P(a)->ce->name;
        if (strcmp(ZSTR_VAL(class_name), "GdImage") == 0) {
            NDArray_FREE(nda);
        }
    }
#endif
}

void RETURN_NDARRAY(NDArray* array, zval* return_value) {
    if (array == NULL) {
        RETURN_THROWS();
        return;
    }
    if (NDArray_NDIM(array) > 0) {
        add_to_buffer(array);
        object_init_ex(return_value, phpsci_ce_NDArray);
        ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(return_value), 0), NDArray_UUID(array));
    } else {
        ZVAL_DOUBLE(return_value, NDArray_GetFloatScalar(array));
        NDArray_FREE(array);
    }
}

NDArray**
ARRAY_OF_NDARRAYS(zval *array, int *size) {
    zval *val;
    NDArray **rtn = NULL;
    int cur_index = 0;
    rtn = emalloc(sizeof(NDArray*) * zend_array_count(Z_ARRVAL_P(array)));
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(array), val) {
        if (Z_TYPE_P(val) == IS_ARRAY) {
            rtn[cur_index] = ZVAL_TO_NDARRAY(val);
        }
        if (Z_TYPE_P(val) == IS_OBJECT) {
            zend_class_entry* ce = NULL;
            ce = Z_OBJCE_P(val);
            if (ce == phpsci_ce_NDArray) {
                rtn[cur_index] = buffer_get(get_object_uuid(val));
            }
        }
        cur_index++;
    } ZEND_HASH_FOREACH_END();
    *size = cur_index;
    return rtn;
}

static int ndarray_objects_compare(zval *obj1, zval *obj2) {
    zval result;
    NDArray *a, *b, *c;

    a = ZVAL_TO_NDARRAY(obj1);
    b = ZVAL_TO_NDARRAY(obj2);

    if (NDArray_ArrayEqual(a, b)) {
        return 0;
    }
    return 1;
}

typedef struct {
    zend_object std;
    int value;
} NDArrayObject;

typedef struct {
    zend_object std;
    int value;
} NumPowerObject;

static int ndarray_do_operation_ex(zend_uchar opcode, zval *result, zval *op1, zval *op2) { /* {{{ */
    NDArray *nda = ZVAL_TO_NDARRAY(op1);
    NDArray *ndb = ZVAL_TO_NDARRAY(op2);
    if (nda == NULL | ndb == NULL) {
        return FAILURE;
    }
    NDArray *rtn = NULL;
    switch(opcode) {
    case ZEND_ADD:
        rtn = NDArray_Add_Float(nda, ndb);
        break;
    case ZEND_SUB:
        rtn = NDArray_Subtract_Float(nda, ndb);
        break;
    case ZEND_MUL:
        rtn = NDArray_Multiply_Float(nda, ndb);
        break;
    case ZEND_DIV:
        rtn = NDArray_Divide_Float(nda, ndb);
        break;
    case ZEND_POW:
        rtn = NDArray_Pow_Float(nda, ndb);
        break;
    case ZEND_MOD:
        rtn = NDArray_Mod_Float(nda, ndb);
        break;
    default:
        return FAILURE;
    }
    CHECK_INPUT_AND_FREE(op1, nda);
    CHECK_INPUT_AND_FREE(op2, ndb);
    RETURN_NDARRAY(rtn, result);
    if (rtn != NULL) {
        return SUCCESS;
    }
    return FAILURE;
}

static int arithmetic_do_operation_ex(zend_uchar opcode, zval *result, zval *op1, zval *op2) { /* {{{ */
    zval retval;
    zval method_name;

    switch(opcode) {
        case ZEND_ADD:
            ZVAL_STRING(&method_name, "__add");
            break;
        case ZEND_SUB:
            ZVAL_STRING(&method_name, "__sub");
            break;
        case ZEND_MUL:
            ZVAL_STRING(&method_name, "__mul");
            break;
        case ZEND_DIV:
            ZVAL_STRING(&method_name, "__div");
            break;
        case ZEND_POW:
            ZVAL_STRING(&method_name, "__pow");
            break;
        case ZEND_MOD:
            ZVAL_STRING(&method_name, "__mod");
            break;
        default:
            return FAILURE;
    }

    zval params[1];
    ZVAL_COPY(&params[0], op2);

    if (call_user_function(NULL, op1, &method_name, &retval, 1, params) == FAILURE) {
        zval_ptr_dtor(&method_name);
        return FAILURE;
    }

    // Copy the result to the result zval
    ZVAL_COPY(result, &retval);

    // Clean up
    zval_ptr_dtor(&method_name);
    zval_ptr_dtor(&retval);
    zval_ptr_dtor(&params[0]);

    return SUCCESS;
}

static
int ndarray_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2) { /* {{{ */
    int retval;
    retval = ndarray_do_operation_ex(opcode, result, op1, op2);
    return retval;
}

static
int arithmetic_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2) { /* {{{ */
    int retval;
    retval = arithmetic_do_operation_ex(opcode, result, op1, op2);
    return retval;
}

static void ndarray_destructor(zend_object* object) {
    zval *obj_uuid = OBJ_PROP_NUM(object, 0);
    if (GC_REFCOUNT(object) <= 1 && Z_TYPE_P(obj_uuid) != IS_UNDEF) {
        buffer_ndarray_free((int)Z_LVAL_P(obj_uuid));
        zend_object_std_dtor(object);
    }
}

static void ndarray_objects_init(zend_class_entry *class_type) {
    memcpy(&ndarray_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    ndarray_object_handlers.compare = ndarray_objects_compare;
    ndarray_object_handlers.do_operation = ndarray_do_operation;
    ndarray_object_handlers.free_obj = ndarray_destructor;
}

static void numpower_objects_init(zend_class_entry *class_type) {
    memcpy(&numpower_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));\
}

static void arithmetic_objects_init(zend_class_entry *class_type) {
    memcpy(&arithmetic_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    arithmetic_object_handlers.do_operation = arithmetic_do_operation;
}

static zend_object *ndarray_create_object(zend_class_entry *class_type) {
    NDArrayObject *intern = zend_object_alloc(sizeof(NDArrayObject), class_type);

    zend_object_std_init(&intern->std, class_type);
    object_properties_init(&intern->std, class_type);
    intern->std.handlers = &ndarray_object_handlers;

    return &intern->std;
}

static zend_object *numpower_create_object(zend_class_entry *class_type) {
    NumPowerObject *intern = zend_object_alloc(sizeof(NumPowerObject), class_type);
    zend_object_std_init(&intern->std, class_type);
    object_properties_init(&intern->std, class_type);
    intern->std.handlers = &numpower_object_handlers;
    return &intern->std;
}

static zend_object *arithmetic_create_object(zend_class_entry *class_type) {
    NDArrayObject *intern = zend_object_alloc(sizeof(NDArrayObject), class_type);
    zend_object_std_init(&intern->std, class_type);
    object_properties_init(&intern->std, class_type);
    intern->std.handlers = &arithmetic_object_handlers;
    return &intern->std;
}

NDArray* ZVALUUID_TO_NDARRAY(zval* obj) {
    if (Z_TYPE_P(obj) == IS_LONG) {
        return buffer_get(Z_LVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_OBJECT) {
        return buffer_get(get_object_uuid(obj));
    }
    return NULL;
}

void RETURN_2NDARRAY(NDArray* array1, NDArray* array2, zval* return_value) {
    zval a, b;
    if (array1 == NULL) {
        RETURN_THROWS();
        return;
    }
    if (array2 == NULL) {
        RETURN_THROWS();
        return;
    }

    add_to_buffer(array1);
    add_to_buffer(array2);

    RETURN_NDARRAY(array1, &a);
    RETURN_NDARRAY(array2, &b);
    array_init(return_value);
    add_next_index_object(return_value, Z_OBJ(a));
    add_next_index_object(return_value, Z_OBJ(b));
}

void RETURN_3NDARRAY(NDArray* array1, NDArray* array2, NDArray* array3, zval* return_value) {
    zval a, b, c;
    if (array1 == NULL) {
        RETURN_THROWS();
        return;
    }
    if (array2 == NULL) {
        RETURN_THROWS();
        return;
    }
    if (array3 == NULL) {
        RETURN_THROWS();
        return;
    }

    add_to_buffer(array1);
    add_to_buffer(array2);
    add_to_buffer(array3);

    object_init_ex(&a, phpsci_ce_NDArray);
    object_init_ex(&b, phpsci_ce_NDArray);
    object_init_ex(&c, phpsci_ce_NDArray);

    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(&a), 0), NDArray_UUID(array1));
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(&b), 0), NDArray_UUID(array2));
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(&c), 0), NDArray_UUID(array3));

    array_init_size(return_value, 3);
    add_next_index_zval(return_value, &a);
    add_next_index_zval(return_value, &b);
    add_next_index_zval(return_value, &c);
    RETURN_ZVAL(return_value, 0, 0);
}

void
bypass_printr() {
    zend_string* functionToRename = zend_string_init("print_r", strlen("print_r"), 0);
    zend_function* functionEntry = zend_hash_find_ptr(EG(function_table), functionToRename);

    if (functionEntry != NULL) {
        zend_string* newFunctionName = zend_string_init("print_r_", strlen("print_r_"), 1);
        zend_string_release_ex(functionEntry->common.function_name, 0);
        functionEntry->common.function_name = newFunctionName;
        functionEntry->internal_function.handler = ZEND_FN(print_r_);
        zend_string_addref(functionEntry->common.function_name);
    }
    zend_string_release_ex(functionToRename, 0);
}

ZEND_BEGIN_ARG_INFO(arginfo_ArithmeticOperand_construct, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(ArithmeticOperand, __construct) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
}

ZEND_BEGIN_ARG_INFO(arginfo_NumPower_construct, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NumPower, __construct) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
}

/* }}}*/
ZEND_BEGIN_ARG_INFO(arginfo_construct, 1)
ZEND_ARG_INFO(0, obj_zval)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, __construct) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *obj_zval;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(obj_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    add_to_buffer(array);
    ZVAL_LONG(OBJ_PROP_NUM(obj, 0), NDArray_UUID(array));
}

ZEND_BEGIN_ARG_INFO(arginfo_fill, 1)
ZEND_ARG_INFO(0, value)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, fill) {
    double value;
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    NDArray_Fill(array, (float)value);
}

ZEND_BEGIN_ARG_INFO(arginfo_toArray, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, toArray) {
    zval rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray must be on CPU RAM before it can be converted to a PHP array.");
        return;
    }
    if (NDArray_NDIM(array) == 0) {
        RETURN_DOUBLE(NDArray_FDATA(array)[0]);
        NDArray_FREE(array);
        return;
    }

    rtn = NDArray_ToPHPArray(array);
    RETURN_ZVAL(&rtn, 0, 0);
}

ZEND_BEGIN_ARG_INFO(arginfo_toImage, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, toImage) {
    zval *alpha = NULL;
    zval *obj_zval = getThis();
    NDArray *n_alpha = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_ZVAL(alpha)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (alpha != NULL) {
        n_alpha = ZVAL_TO_NDARRAY(alpha);
    }
    if (array == NULL) {
        return;
    }
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray must be on CPU RAM before it can be converted to a GD image.");
        return;
    }
    if (NDArray_NDIM(array) != 3 || NDArray_SHAPE(array)[0] != 3) {
        zend_throw_error(NULL, "NDArray must be 3-dimensional before it can be converted to a RGB image.");
        return;
    }
    NDArray_ToGD(array, n_alpha, return_value);
    if (alpha != NULL) {
        CHECK_INPUT_AND_FREE(alpha, n_alpha);
    }
}

ZEND_BEGIN_ARG_INFO(arginfo_gpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, gpu) {
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
#ifdef HAVE_CUBLAS
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    rtn = NDArray_ToGPU(array);
    RETURN_NDARRAY(rtn, return_value);
#else
    zend_throw_error(NULL, "No GPU device available or CUDA not enabled");
    RETURN_NULL();
#endif
}

ZEND_BEGIN_ARG_INFO(arginfo_cpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, cpu) {
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    rtn = NDArray_ToCPU(array);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_is_gpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, isGPU) {
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);

    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
        RETURN_LONG(0);
    } else {
        RETURN_LONG(1);
    }
}

ZEND_BEGIN_ARG_INFO(arginfo_dump, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, dump) {
    zval rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    NDArray_Dump(array);
}

ZEND_BEGIN_ARG_INFO(arginfo_dump_devices, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NumPower, dumpDevices) {
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray_DumpDevices();
}

ZEND_BEGIN_ARG_INFO(arginfo_load, 0)
    ZEND_ARG_INFO(0, name)
ZEND_END_ARG_INFO();
PHP_METHOD(NumPower, load) {
    zend_string *name;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *rtn = NDArray_Load(name->val);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_save, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, name)
ZEND_END_ARG_INFO();
PHP_METHOD(NumPower, save) {
    zval *a;
    zend_string *name;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *array = ZVAL_TO_NDARRAY(a);
    NDArray_Save(array, name->val, name->len);
    CHECK_INPUT_AND_FREE(a, array);
    RETURN_NULL();
}

ZEND_BEGIN_ARG_INFO(arginfo_setdevice, 0)
ZEND_ARG_INFO(0, deviceId)
ZEND_END_ARG_INFO();
PHP_METHOD(NumPower, setDevice) {
    int numDevices;
    long deviceId;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(deviceId)
    ZEND_PARSE_PARAMETERS_END();
#ifdef HAVE_CUBLAS
    // Get the number of available CUDA devices
    cudaError_t cudaError = cudaGetDeviceCount(&numDevices);

    if (cudaError != cudaSuccess) {
        zend_throw_error(NULL, "Error getting the number of CUDA devices.\n");
        return;
    }
    if (deviceId >= 0 && deviceId > (numDevices - 1)) {
        zend_throw_error(NULL, "Device %d does not exist.\n", (int)deviceId);
        return;
    }
    cudaSetDevice(deviceId);
#endif
}

// @todo Indices conversion lose precision, we must convert it directly to a integer vector in C
//       without relying on ZVAL_TO_NDARRAY. We must apply the same for all other cases where a
//       PHP array of longs is converted to NDArray before being converted to a C integer.
ZEND_BEGIN_ARG_INFO(arginfo_reshape, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, reshape) {
    int *new_shape;
    zval *shape_zval;
    zval *a;
    NDArray *rtn;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* target = ZVAL_TO_NDARRAY(a);
    NDArray* shape = ZVAL_TO_NDARRAY(shape_zval);
    new_shape = NDArray_ToIntVector(shape);

    rtn = NDArray_Reshape(target, new_shape, NDArray_NUMELEMENTS(shape));

    if (rtn == NULL) {
        NDArray_FREE(shape);
        efree(new_shape);
        RETURN_NULL();
    }

    if (Z_TYPE_P(shape_zval) == IS_ARRAY) {
        NDArray_FREE(shape);
    }
    CHECK_INPUT_AND_FREE(a, target);
    RETURN_NDARRAY(rtn, return_value);
}

PHP_FUNCTION(print_r_) {
    zval *var;
    bool do_return = 0;
    NDArray *target;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(var)
    Z_PARAM_OPTIONAL
    Z_PARAM_BOOL(do_return)
    ZEND_PARSE_PARAMETERS_END();

    if (do_return) {
        if (Z_TYPE_P(var) == IS_OBJECT) {
            zend_class_entry* classEntry = Z_OBJCE_P(var);
            if (!strcmp(classEntry->name->val, "NDArray")) {
                target = buffer_get(get_object_uuid(var));
                RETURN_STRING(NDArray_Print(target, 1));
            }
        }
        RETURN_STR(zend_print_zval_r_to_str(var, 0));
    } else {
        if (Z_TYPE_P(var) == IS_OBJECT) {
            zend_class_entry* classEntry = Z_OBJCE_P(var);
            if (!strcmp(classEntry->name->val, "NDArray")) {
                target = buffer_get(get_object_uuid(var));
                NDArray_Print(target, 0);
                RETURN_TRUE;
            }
        }
        zend_print_zval_r(var, 0);
        RETURN_TRUE;
    }
}

/**
 * NumPower::zeros
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_zeros, 1)
ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, zeros) {
    NDArray *rtn = NULL;
    int *shape;
    zval *shape_zval;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(shape_zval);
    if (nda == NULL) {
        return;
    }
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
        shape[i] = (int) NDArray_FDATA(nda)[i];
    }
    rtn = NDArray_Zeros(shape, NDArray_NUMELEMENTS(nda), NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::equal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_equal, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, equal) {
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_Equal(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::greater
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_greater, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, greater) {
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_Greater(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::greaterEqual
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_greaterequal, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, greaterEqual) {
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_GreaterEqual(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::less
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_less, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, less) {
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_Less(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::lessEqual
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_lessequal, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, lessEqual) {
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_LessEqual(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::notEqual
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_notequal, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, notEqual) {
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_NotEqual(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::identity
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_identity, 1)
ZEND_ARG_INFO(0, size)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, identity) {
    NDArray *rtn = NULL;
    int *shape;
    long size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();
    rtn = NDArray_Identity((int)size);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::normal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_normal, 0, 0, 1)
    ZEND_ARG_INFO(0, size)
    ZEND_ARG_INFO(0, loc)
    ZEND_ARG_INFO(0, scale)
    ZEND_ARG_INFO(0, accelerator)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, normal) {
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    long accelerator = NDARRAY_DEVICE_CPU;
    double loc = 0.0, scale = 1.0;
    
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(size)
    Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(loc)
        Z_PARAM_DOUBLE(scale)
        Z_PARAM_LONG(accelerator)
    ZEND_PARSE_PARAMETERS_END();
    
    int accelerator_i = (int) accelerator;
    NDArray *nda = ZVAL_TO_NDARRAY(size);
    
    if (nda == NULL) return;
    
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
        shape[i] = (int) NDArray_FDATA(nda)[i];
    }

    rtn = NDArray_Normal(loc, scale, shape, NDArray_NUMELEMENTS(nda), accelerator_i);
    NDArray_FREE(nda);
    
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::truncatedNormal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_truncated_normal, 0, 0, 1)
    ZEND_ARG_INFO(0, size)
    ZEND_ARG_INFO(0, loc)
    ZEND_ARG_INFO(0, scale)
    ZEND_ARG_INFO(0, accelerator)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, truncatedNormal) {
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    long accelerator = NDARRAY_DEVICE_CPU;
    double loc = 0.0, scale = 1.0;

    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(size)
    Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(loc)
        Z_PARAM_DOUBLE(scale)
        Z_PARAM_LONG(accelerator)
    ZEND_PARSE_PARAMETERS_END();

    int accelerator_i = (int) accelerator;
    NDArray *nda = ZVAL_TO_NDARRAY(size);

    if (nda == NULL) {
        return;
    }

    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));

    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
        shape[i] = (int) NDArray_FDATA(nda)[i];
    }

    rtn = NDArray_TruncatedNormal(loc, scale, shape, NDArray_NUMELEMENTS(nda), accelerator_i);
    NDArray_FREE(nda);

    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::randomBinomial
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_binomial, 0, 0, 3)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, p)
    ZEND_ARG_INFO(0, n)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, randomBinomial) {
    NDArray *rtn = NULL;
    int *ishape;
    zval* shape;
    double n = 0.0, p = 1.0;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(shape)
        Z_PARAM_DOUBLE(n)
        Z_PARAM_DOUBLE(p)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(shape);
    if (nda == NULL) return;
    ishape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
        ishape[i] = (int) NDArray_FDATA(nda)[i];
    }
    rtn = NDArray_Binomial(ishape, NDArray_NUMELEMENTS(nda), (int)n, (float)p);
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::standardNormal
 *
 * @param shape
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_standard_normal, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(NumPower, standardNormal) {
    NDArray *rtn = NULL;
    zval* shape;
    HashTable *shape_ht;
    zend_string *key;
    zend_ulong idx;
    zval *val;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape)
    ZEND_PARSE_PARAMETERS_END();

    shape_ht = Z_ARRVAL_P(shape);

    ZEND_HASH_FOREACH_KEY_VAL(shape_ht, idx, key, val) {
        if (Z_TYPE_P(val) != IS_LONG) {
            zend_throw_error(NULL, "Invalid parameter: Shape elements must be integers.");
            return;
        }
    } ZEND_HASH_FOREACH_END();

    NDArray *nda = ZVAL_TO_NDARRAY(shape);

    if (nda == NULL) {
        return;
    }

    if (NDArray_NUMELEMENTS(nda) == 0) {
        NDArray_FREE(nda);
        zend_throw_error(NULL, "Invalid parameter: Expected a non-empty array.");
        return;
    }

    rtn = NDArray_StandardNormal(NDArray_ToIntVector(nda), NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);

    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::poisson
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_poisson, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO(0, lam, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, poisson) {
    NDArray *rtn = NULL;
    zval* shape;
    HashTable *shape_ht;
    zend_string *key;
    zend_ulong idx;
    zval *val;
    double lam = 1.0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(shape)
    Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(lam)
    ZEND_PARSE_PARAMETERS_END();

    shape_ht = Z_ARRVAL_P(shape);

    ZEND_HASH_FOREACH_KEY_VAL(shape_ht, idx, key, val) {
        if (Z_TYPE_P(val) != IS_LONG) {
            zend_throw_error(NULL, "Invalid parameter: Shape elements must be integers.");
            return;
        }
    } ZEND_HASH_FOREACH_END();

    NDArray *nda = ZVAL_TO_NDARRAY(shape);

    if (nda == NULL) {
        return;
    }

    if (NDArray_NUMELEMENTS(nda) == 0) {
        NDArray_FREE(nda);
        zend_throw_error(NULL, "Invalid parameter: Expected a non-empty array.");
        return;
    }

    rtn = NDArray_Poisson(lam, NDArray_ToIntVector(nda), NDArray_NUMELEMENTS(nda));

    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::uniform
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_uniform, 0, 0, 1)
ZEND_ARG_INFO(0, size)
ZEND_ARG_INFO(0, low)
ZEND_ARG_INFO(0, high)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, uniform) {
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    double low = 0.0, high = 1.0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_ZVAL(size)
    Z_PARAM_OPTIONAL
    Z_PARAM_DOUBLE(low)
    Z_PARAM_DOUBLE(high)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(size);
    if (nda == NULL) {
        return;
    }
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
        shape[i] = (int) NDArray_FDATA(nda)[i];
    }
    rtn = NDArray_Uniform(low, high, shape, NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::diag
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_diag, 0, 0, 1)
ZEND_ARG_INFO(0, target)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, diag) {
    NDArray *rtn = NULL;
    zval* target;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(target)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(target);
    if (nda == NULL)  return;
    rtn = NDArray_Diag(nda);
    if (Z_TYPE_P(target) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::diagonal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_diagonal, 0, 0, 1)
ZEND_ARG_INFO(0, target)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, diagonal) {
    NDArray *rtn = NULL;
    zval* target;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(target)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(target);
    if (nda == NULL)  return;
    rtn = NDArray_Diagonal(nda, 0);
    if (Z_TYPE_P(target) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::full
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_full, 0, 0, 2)
ZEND_ARG_INFO(0, shape)
ZEND_ARG_INFO(0, fill_value)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, full) {
    NDArray *rtn = NULL;
    zval* shape;
    HashTable *shape_ht;
    double fill_value;
    int *new_shape;
    zend_string *key;
    zend_ulong idx;
    zval *val;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_DOUBLE(fill_value)
    ZEND_PARSE_PARAMETERS_END();
    shape_ht = Z_ARRVAL_P(shape);

    ZEND_HASH_FOREACH_KEY_VAL(shape_ht, idx, key, val) {
        if (Z_TYPE_P(val) != IS_LONG) {
            zend_throw_error(NULL, "Invalid parameter: Shape elements must be integers.");
            return;
        }
    } ZEND_HASH_FOREACH_END();

    NDArray *nd_shape = ZVAL_TO_NDARRAY(shape);

    if (nd_shape == NULL) {
        return;
    }

    if (NDArray_NUMELEMENTS(nd_shape) == 0) {
        NDArray_FREE(nd_shape);
        zend_throw_error(NULL, "Invalid parameter: Expected a non-empty array.");
        return;
    }

    new_shape = NDArray_ToIntVector(nd_shape);
    rtn = NDArray_Full(new_shape, NDArray_NUMELEMENTS(nd_shape), fill_value);

    efree(new_shape);
    NDArray_FREE(nd_shape);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::ones
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_ones, 1)
ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, ones) {
    double *ptr;
    NDArray *rtn = NULL;
    int *shape;
    zval *shape_zval;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(shape_zval);
    if (nda == NULL) {
        return;
    }
    shape = NDArray_ToIntVector(nda);
    rtn = NDArray_Ones(shape, NDArray_NUMELEMENTS(nda), NDARRAY_TYPE_FLOAT32);
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arange
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arange, 0, 0, 1)
ZEND_ARG_INFO(0, stop)
ZEND_ARG_INFO(0, start)
ZEND_ARG_INFO(0, step)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arange) {
    NDArray *rtn = NULL;
    double start, stop, step;
    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_DOUBLE(stop)
    Z_PARAM_OPTIONAL
    Z_PARAM_DOUBLE(start)
    Z_PARAM_DOUBLE(step)
    ZEND_PARSE_PARAMETERS_END();
    if (ZEND_NUM_ARGS() == 1) {
        start = 0.0f;
        step  = 1.0f;
    }
    if (ZEND_NUM_ARGS() == 2) {
        step  = 1.0f;
    }
    rtn = NDArray_Arange(start, stop, step);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::all
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_all, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, all) {
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(array)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        RETURN_LONG(NDArray_All(nda));
        if (Z_TYPE_P(array) == IS_ARRAY) {
            NDArray_FREE(nda);
        }
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
        rtn = single_reduce(nda, &axis_i, &NDArray_All);
        RETURN_NDARRAY(rtn, return_value);
    }
}

/**
 * NumPower::allClose
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_allclose, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_ARG_INFO(0, rtol)
ZEND_ARG_INFO(0, atol)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, allClose) {
    zval *a, *b;
    double rtol = 1e-05, atol = 1e-08;
    int rtn;
    ZEND_PARSE_PARAMETERS_START(2, 4)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    Z_PARAM_OPTIONAL
    Z_PARAM_DOUBLE(rtol)
    Z_PARAM_DOUBLE(atol)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);

    if (nda == ndb) {
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_BOOL(true);
    }

    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_AllClose(nda, ndb, (float)rtol, (float)atol);
    if (rtn == -1) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_BOOL(rtn);
}

/**
 * NumPower::transpose
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_transpose, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axes)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, transpose) {
    NDArray *rtn = NULL;
    zval *array, *axes;
    HashTable *axes_ht;
    zend_string *key;
    zend_ulong idx;
    zval *val;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(array)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(axes)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    NDArray_Dims *dims = NULL;
    if (ZEND_NUM_ARGS() == 2) {
        axes_ht = Z_ARRVAL_P(axes);
        dims = emalloc(sizeof(NDArray_Dims));
        dims->len = (int)zend_array_count(axes_ht);
        dims->ptr = emalloc(sizeof(int) * dims->len);
        ZEND_HASH_FOREACH_KEY_VAL(axes_ht, idx, key, val)
        {
            if (Z_TYPE_P(val) != IS_LONG) {
                zend_throw_error(NULL, "Invalid parameter: axes elements must be integers.");
                return;
            }
            dims->ptr[(int)idx] = (int)zval_get_long(val);
        }ZEND_HASH_FOREACH_END();
    }

    rtn = NDArray_Transpose(nda, dims);
    CHECK_INPUT_AND_FREE(array, nda);
    if (ZEND_NUM_ARGS() == 2) efree(dims);
    if (rtn == NULL) return;
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::copy
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_copy, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, device)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, copy) {
    NDArray *rtn = NULL;
    zval *array;
    long device = -1;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(array)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (device == -1) {
        device = NDArray_DEVICE(nda);
    }
    if (device != NDARRAY_DEVICE_CPU && device != NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "$device argument must be either 0 (CPU) or 1 (GPU)");
        CHECK_INPUT_AND_FREE(array, nda);
        return;
    }
    rtn = NDArray_Copy(nda, NDArray_DEVICE(nda));
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::atleast1d
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_atleast_1d, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, atleast1d) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    NDArray *output = NDArray_AtLeast1D(nda);

    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(output, return_value);
}

/**
 * NumPower::atleast2d
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_atleast_2d, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, atleast2d) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    NDArray *output = NDArray_AtLeast2D(nda);

    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(output, return_value);
}

/**
 * NumPower::atleast3d
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_atleast_3d, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, atleast3d) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    NDArray *output = NDArray_AtLeast3D(nda);

    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(output, return_value);
}

/**
 * NDArray::shape
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_shape, 0, 0, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, shape) {
    NDArray *rtn = NULL;
    zval *array = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);

    array_init_size(return_value, NDArray_NDIM(nda));
    for (int i = 0; i < NDArray_NDIM(nda); i++) {
        add_index_long(return_value, i, NDArray_SHAPE(nda)[i]);
    }
}

/**
 * NumPower::flatten
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_flat, 0, 0, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, flatten) {
    NDArray *rtn = NULL;
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Flatten(nda);
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::abs
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_abs, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, abs) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Abs(nda);

    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sin, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sin) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sin);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sin);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::cos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_cos, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, cos) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_cos);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_cos);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_tan, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, tan) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_tan);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_tan);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arcsin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arcsin, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arcsin) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arcsin);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arcsin);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::rsqrt
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_rsqrt, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, rsqrt) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_rsqrt);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arccos);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arccos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arccos, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arccos) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arccos);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arccos);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arccos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctan, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arctan) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arctan);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arctan);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arctan2
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctan2, 0, 0, 2)
    ZEND_ARG_INFO(0, x)
    ZEND_ARG_INFO(0, y)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arctan2) {
    NDArray *rtn = NULL;
    zval *x, *y;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x)
            Z_PARAM_ZVAL(y)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *ndx = ZVAL_TO_NDARRAY(x);
    NDArray *ndy = ZVAL_TO_NDARRAY(y);
    if (x == NULL || y == NULL) {
        return;
    }

    if (NDArray_DEVICE(ndx) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map1ND(ndx, float_arctan2, ndy);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise1N(ndx, cuda_float_arctan2, ndy);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(x, ndx);
    CHECK_INPUT_AND_FREE(y, ndy);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::degrees
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_degrees, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, degrees) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_degrees);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_degrees);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sinh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sinh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sinh) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sinh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sinh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::cosh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_cosh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, cosh) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_cosh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_cosh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::tanh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_tanh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, tanh) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_tanh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_tanh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arcsinh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arcsinh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arcsinh) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arcsinh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arcsinh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arccosh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arccosh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arccosh) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arccosh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arccosh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::fromImage
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_fromimage, 0, 0, 1)
    ZEND_ARG_INFO(0, image)
    ZEND_ARG_INFO(0, channelLast)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, fromImage) {
    NDArray *rtn = NULL;
    zval *image;
    bool channelLast = true;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(image)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(channelLast)
    ZEND_PARSE_PARAMETERS_END();
    rtn = NDArray_FromGD(image, channelLast);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arctanh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctanh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arctanh) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arctanh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arctanh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::rint
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_rint, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, rint) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_rint);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_rint);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::fix
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_fix, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, fix) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_fix);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_fix);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::trunc
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_trunc, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, trunc) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_trunc);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_trunc);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sinc
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sinc, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sinc) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sinc);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sinc);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::negative
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_negative, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, negative) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_negate);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_negate);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::positive
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_positive, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, positive) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_positive);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_positive);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::reciprocal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_reciprocal, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, reciprocal) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_reciprocal);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_reciprocal);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}


/**
 * NumPower::sign
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sign, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sign) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sign);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sign);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::clip
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_clip, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, min)
ZEND_ARG_INFO(0, max)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, clip) {
    NDArray *rtn = NULL;
    zval *array;
    double min, max;
    ZEND_PARSE_PARAMETERS_START(3, 3)
    Z_PARAM_ZVAL(array)
    Z_PARAM_DOUBLE(min)
    Z_PARAM_DOUBLE(max)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map2F(nda, float_clip, (float)min, (float)max);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise2F(nda, cuda_float_clip, (float)min, (float)max);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    if (Z_TYPE_P(array) != IS_ARRAY) {
        CHECK_INPUT_AND_FREE(array, nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::maximum
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_maximum, 0, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, maximum) {
    NDArray *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL || ndb == NULL) {
        return;
    }

    rtn = NDArray_Maximum(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::minimum
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_minimum, 0, 0, 1)
        ZEND_ARG_INFO(0, a)
        ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, minimum) {
    NDArray *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL || ndb == NULL) {
        return;
    }

    rtn = NDArray_Minimum(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}


/**
 * NumPower::argmax
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_argmax, 0, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
    ZEND_ARG_INFO(0, keepdims)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, argmax) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    bool keepdims = false;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
        Z_PARAM_BOOL(keepdims)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 1) {
        axis = 128;
    }
    rtn = NDArray_ArgMinMaxCommon(nda, (int)axis, keepdims, true);
    if (rtn == NULL) return;
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::argmin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_argmin, 0, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
    ZEND_ARG_INFO(0, keepdims)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, argmin) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    bool keepdims = false;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
        Z_PARAM_BOOL(keepdims)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 1) {
        axis = 128;
    }
    rtn = NDArray_ArgMinMaxCommon(nda, (int)axis, keepdims, false);
    if (rtn == NULL) return;
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::mean
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_mean, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, mean) {
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(array)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            NDArray *sum = reduce(nda, &i_axis, NDArray_Add_Float);
            if (sum == NULL) {
                CHECK_INPUT_AND_FREE(array, nda);
                return;
            }
            NDArray *num_cols = NDArray_CreateFromLongScalar((long)NDArray_SHAPE(nda)[i_axis]);
            rtn = NDArray_Divide_Float(sum, num_cols);
            NDArray_FREE(sum);
            NDArray_FREE(num_cols);
        }
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::median
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_median, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, median) {
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        RETURN_DOUBLE(NDArray_Median_Float(nda));
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE(NDArray_Median_Float(nda));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Median_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::std
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_std, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, std) {
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Std(nda);
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            rtn = NDArray_Std(nda);
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (rtn == NULL) {
        return;
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::quantile
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_quantile, 0, 0, 1)
ZEND_ARG_INFO(0, target)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, quantile) {
    NDArray *rtn = NULL;
    zval *a, *q;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(q)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndq = ZVAL_TO_NDARRAY(q);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Quantile(nda, ndq);

    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(q, ndq);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::std
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_average, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, weights)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, average) {
    NDArray *rtn = NULL;
    zval *array, *weights = NULL;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(array)
    Z_PARAM_OPTIONAL
    Z_PARAM_ZVAL(weights)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Average(nda, NULL);
    }
    if (ZEND_NUM_ARGS() == 2) {
        NDArray *ndw = ZVAL_TO_NDARRAY(weights);
        rtn = NDArray_Average(nda, ndw);
        CHECK_INPUT_AND_FREE(weights, ndw);
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::variance
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_variance, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, variance) {
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Variance(nda);
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            rtn = NDArray_Variance(nda);
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (rtn == NULL) {
        return;
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::ceil
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_ceil, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, ceil) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_ceil);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_ceil);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::round
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_round, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_ARG_INFO(0, precision)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, round) {
    NDArray *rtn = NULL;
    zval *array;
    long precision;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(array)
    Z_PARAM_LONG(precision)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map1F(nda, float_round, (float)precision);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise1F(nda, cuda_float_round, (float)precision);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::floor
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_floor, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, floor) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_floor);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_floor);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::arccos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_radians, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, radians) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_radians);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_radians);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sqrt
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sqrt, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sqrt) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sqrt);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sqrt);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sqrt
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_square, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, square) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Multiply_Float(nda, nda);
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::exp
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_exp, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, exp) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_exp);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_exp);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::exp2
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_exp2, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, exp2) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Map(nda, float_exp2);
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::exp2
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_expm1, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, expm1) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_expm1);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_expm1);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::log
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::logb
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_logb, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, logb) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_logb);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_logb);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::log10
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log10, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log10) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log10);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log10);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::log1p
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log1p, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log1p) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log1p);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log1p);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::log1p
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log2, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log2) {
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log2);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log2);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::subtract
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_subtract, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, subtract) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
    }
    rtn = NDArray_Subtract_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::mod
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_mod, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, mod) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
    }
    rtn = NDArray_Mod_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(a, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::pow
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_pow, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, pow) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_ShapeCompare(nda, ndb)) {
        zend_throw_error(NULL, "Incompatible shapes");
        return;
    }
    rtn = NDArray_Pow_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::multiply
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_multiply, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, multiply) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
    }
    rtn = NDArray_Multiply_Float(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::divide
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_divide, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, divide) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        return;
    }
    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
    }
    rtn = NDArray_Divide_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::add
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_add, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, add) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
    }
    rtn = NDArray_Add_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::expandDims
*/
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_expand_dims, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, expandDims) {
    NDArray *rtn = NULL;
    zval *a;
    zval *axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(axis)
    ZEND_PARSE_PARAMETERS_END();

    if (Z_TYPE_P(axis) != IS_ARRAY && Z_TYPE_P(axis) != IS_LONG && Z_TYPE_P(axis) != IS_OBJECT) {
        zend_throw_error(NULL, "expected array, integer or ndarray");
        return;
    }
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndaxis = ZVAL_TO_NDARRAY(axis);
    if (nda == NULL || ndaxis == NULL) {
        return;
    }
    rtn = NDArray_ExpandDim(nda, ndaxis);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(axis, ndaxis);
    if (rtn == NULL) {
        return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::squeeze
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_squeeze, 0, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, squeeze) {
    NDArray *rtn = NULL, *ndaxis = NULL;
    zval *a;
    zval *axis;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(axis)
    ZEND_PARSE_PARAMETERS_END();

    if (Z_TYPE_P(axis) != IS_ARRAY && Z_TYPE_P(axis) != IS_LONG && Z_TYPE_P(axis) != IS_OBJECT && ZEND_NUM_ARGS() > 1) {
        zend_throw_error(NULL, "expected array, integer or ndarray");
        return;
    }
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (ZEND_NUM_ARGS() > 1) {
        ndaxis = ZVAL_TO_NDARRAY(axis);
    }
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Squeeze(nda, ndaxis);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(axis, ndaxis);
    if (rtn == NULL) {
        return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::flip
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_flip, 0, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, flip) {
    NDArray *rtn = NULL;
    zval *a;
    zval *axis = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(axis)
    ZEND_PARSE_PARAMETERS_END();

    if (Z_TYPE_P(axis) != IS_ARRAY && Z_TYPE_P(axis) != IS_LONG && Z_TYPE_P(axis) != IS_OBJECT) {
        zend_throw_error(NULL, "expected array, integer or ndarray");
        return;
    }
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndaxis = ZVAL_TO_NDARRAY(axis);
    if (nda == NULL || ndaxis == NULL) {
        return;
    }
    rtn = NDArray_Flip(nda, ndaxis);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(axis, ndaxis);
    if (rtn == NULL) {
        return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::swapAxes
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_swapaxes, 0, 0, 3)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis1)
    ZEND_ARG_INFO(0, axis2)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, swapAxes) {
    NDArray *rtn = NULL;
    zval *a;
    long axis1, axis2;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_LONG(axis1)
        Z_PARAM_LONG(axis2)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_SwapAxes(nda, (int) axis1, (int) axis2);

    CHECK_INPUT_AND_FREE(a, nda);
    if (rtn == NULL) {
        return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::rollAxis
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_rollaxis, 0, 0, 2)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
    ZEND_ARG_INFO(0, start)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, rollAxis) {
    NDArray *rtn = NULL;
    zval *a;
    long axis, start = 0;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_LONG(axis)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(start)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
    return;
    }

    rtn = NDArray_Rollaxis(nda, (int) axis, (int) start);

    CHECK_INPUT_AND_FREE(a, nda);
    if (rtn == NULL) {
    return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::moveAxis
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_moveaxis, 0, 0, 3)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, source)
ZEND_ARG_INFO(0, destination)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, moveAxis) {
    NDArray *rtn = NULL;
    zval *a;
    zval *source, *destination;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(source)
        Z_PARAM_ZVAL(destination)
    ZEND_PARSE_PARAMETERS_END();
    int src_size, dest_size;
    int *src = zval_axis_argument(source, "source", &src_size);
    int *dest = zval_axis_argument(destination, "destination", &dest_size);
    if (src == NULL) return;
    if (dest == NULL) return;

    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Moveaxis(nda, src, dest, src_size, dest_size);

    CHECK_INPUT_AND_FREE(a, nda);
    efree(src);
    efree(dest);
    if (rtn == NULL) {
        return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::verticalStack
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_vstack, 0, 0, 1)
    ZEND_ARG_INFO(0, arrays)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, verticalStack) {
    NDArray *rtn = NULL;
    zval *arrays;
    int num_args;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(arrays)
    ZEND_PARSE_PARAMETERS_END();
    NDArray **ndarrays = ARRAY_OF_NDARRAYS(arrays, &num_args);
    if (ndarrays == NULL) return;
    rtn = NDArray_VSTACK(ndarrays, num_args);

    for (int i = 0; i < num_args; i++) {
        NDArray_FREE(ndarrays[i]);
    }
    efree(ndarrays);
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::horizontalStack
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_hstack, 0, 0, 1)
    ZEND_ARG_INFO(0, arrays)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, horizontalStack) {
    NDArray *rtn = NULL;
    zval *arrays;
    int num_args;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(arrays)
    ZEND_PARSE_PARAMETERS_END();
    NDArray **ndarrays = ARRAY_OF_NDARRAYS(arrays, &num_args);
    if (ndarrays == NULL) return;
    rtn = NDArray_HSTACK(ndarrays, num_args);

    for (int i = 0; i < num_args; i++) {
        NDArray_FREE(ndarrays[i]);
    }
    efree(ndarrays);
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::depthStack
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_dstack, 0, 0, 1)
    ZEND_ARG_INFO(0, arrays)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, depthStack) {
    NDArray *rtn = NULL;
    zval *arrays;
    int num_args;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(arrays)
    ZEND_PARSE_PARAMETERS_END();
    NDArray **ndarrays = ARRAY_OF_NDARRAYS(arrays, &num_args);
    if (ndarrays == NULL) return;
    rtn = NDArray_DSTACK(ndarrays, num_args);

    for (int i = 0; i < num_args; i++) {
        NDArray_FREE(ndarrays[i]);
    }
    efree(ndarrays);
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::columnStack
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_column_stack, 0, 0, 1)
    ZEND_ARG_INFO(0, arrays)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, columnStack) {
    NDArray *rtn = NULL;
    zval *arrays;
    int num_args;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(arrays)
    ZEND_PARSE_PARAMETERS_END();
    NDArray **ndarrays = ARRAY_OF_NDARRAYS(arrays, &num_args);
    if (ndarrays == NULL) return;
    rtn = NDArray_ColumnStack(ndarrays, num_args);

    for (int i = 0; i < num_args; i++) {
        NDArray_FREE(ndarrays[i]);
    }
    efree(ndarrays);
    RETURN_NDARRAY(rtn, return_value);
}

/**
* NumPower::concatenate
*/
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_concatenate, 0, 0, 1)
    ZEND_ARG_INFO(0, arrays)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, concatenate) {
    NDArray *rtn = NULL;
    zval *arrays;
    int num_args;
    zval *axis = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(arrays)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray **ndarrays = ARRAY_OF_NDARRAYS(arrays, &num_args);
    if (ndarrays == NULL) return;

    if (ZEND_NUM_ARGS() > 1 && Z_TYPE_P(axis) == IS_NULL) {
        rtn = NDArray_ConcatenateFlat(ndarrays, num_args);
    } else {
        if (ZEND_NUM_ARGS() == 1) {
            rtn = NDArray_Concatenate(ndarrays, num_args, 0);
        } else {
            rtn = NDArray_Concatenate(ndarrays, num_args, zval_get_long(axis));
        }
    }

    for (int i = 0; i < num_args; i++) {
        NDArray_FREE(ndarrays[i]);
    }
    efree(ndarrays);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::append
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_append, 0, 0, 2)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, values)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
    PHP_METHOD(NumPower, append) {
    NDArray *rtn = NULL;
    int num_args;
    zval *axis = NULL, *array, *values;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ZVAL(array)
        Z_PARAM_ZVAL(values)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(axis)
    ZEND_PARSE_PARAMETERS_END();

    NDArray **ndarrays = (NDArray**)emalloc(sizeof(NDArray*) * 2);
    ndarrays[0] = ZVAL_TO_NDARRAY(array);
    ndarrays[1] = ZVAL_TO_NDARRAY(values);
    num_args = 2;
    if (ndarrays == NULL) return;

    if (ZEND_NUM_ARGS() == 2) {
        rtn = NDArray_ConcatenateFlat(ndarrays, num_args);
    } else {
        rtn = NDArray_Concatenate(ndarrays, num_args, zval_get_long(axis));
    }
    CHECK_INPUT_AND_FREE(array, ndarrays[0]);
    CHECK_INPUT_AND_FREE(values, ndarrays[1]);
    efree(ndarrays);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::matmul
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_matmul, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, matmul) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Matmul(nda, ndb);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::inner
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_inner, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, inner) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Inner(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::outer
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_outer, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, outer) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Outer(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::dot
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_dot, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, dot) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Dot(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::trace
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_trace, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, trace) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Trace(nda);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::eig
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_eig, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, eig) {
    NDArray **rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Eig(nda);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_2NDARRAY(rtn[0], rtn[1], return_value);
    efree(rtn);
}

/**
 * NumPower::cholesky
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_cholesky, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, cholesky) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Cholesky(nda);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::solve
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_solve, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, solve) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Solve(nda, ndb);

    if (rtn == NULL) {
        return;
    }

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::lstsq
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_lstsq, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, lstsq) {
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Lstsq(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::qr
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_qr, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, qr) {
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtns = NDArray_Qr(nda);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_2NDARRAY(rtns[0], rtns[1], return_value);
    efree(rtns);
}

/**
 * NumPower::lu
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_lu, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, lu) {
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtns = NDArray_LU(nda);
    if (rtns == NULL) {
        return;
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
}

/**
 * NumPower::matrixRank
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_matrix_rank, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, matrixRank) {
    NDArray *rtn;
    zval *a, *b;
    long axis;
    double tol = 1e-6;
    float tol_p;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
    Z_PARAM_DOUBLE(tol)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    tol_p = (float)tol;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_MatrixRank(nda, NULL);
    } else {
        rtn = NDArray_MatrixRank(nda, &tol_p);
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}


/**
 * NumPower::dnnConv2dForward
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_dnn_conv2d_forward, 4, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, dnnConv2dForward) {
    NDArray *rtn;
    NDArray *ndb = NULL;
    zval *input, *filters;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(input)
        Z_PARAM_ZVAL(filters)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *ndinput = ZVAL_TO_NDARRAY(input);
    NDArray *ndfilters = ZVAL_TO_NDARRAY(filters);

    rtn = NDArrayDNN_Conv2D_Forward(ndinput, ndfilters, NULL, 'r', 1);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(input, ndinput);
    CHECK_INPUT_AND_FREE(filters, ndfilters);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::dnnConv1dForward
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_dnn_conv1d_forward, 4, 0, 1)
                ZEND_ARG_INFO(0, a)
                ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, dnnConv1dForward) {
    NDArray *rtn;
    NDArray *ndb = NULL;
    zval *input, *filters;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(input)
            Z_PARAM_ZVAL(filters)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *ndinput = ZVAL_TO_NDARRAY(input);
    NDArray *ndfilters = ZVAL_TO_NDARRAY(filters);

    rtn = NDArray_DNN_Conv1D(ndinput, ndfilters);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(input, ndinput);
    CHECK_INPUT_AND_FREE(filters, ndfilters);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::dnnConv2dBackward
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_dnn_conv2d_backward, 3, 0, 1)
ZEND_ARG_INFO(0, x)
ZEND_ARG_INFO(0, y)
ZEND_ARG_INFO(0, filters)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, dnnConv2dBackward) {
    NDArray **rtn;
    zval *x, *y, *filters;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(x)
        Z_PARAM_ZVAL(y)
        Z_PARAM_ZVAL(filters)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *ndx = ZVAL_TO_NDARRAY(x);
    NDArray *ndy = ZVAL_TO_NDARRAY(y);
    NDArray *ndfilters = ZVAL_TO_NDARRAY(filters);

    rtn = NDArrayDNN_Conv2D_Backward(ndx, ndy, ndfilters, 3, 'r', 1);

    CHECK_INPUT_AND_FREE(x, ndx);
    CHECK_INPUT_AND_FREE(y, ndy);
    CHECK_INPUT_AND_FREE(filters, ndfilters);
    RETURN_2NDARRAY(rtn[0], rtn[1], return_value);
    efree(rtn);
}

/**
 * NumPower::convolve2d
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_convolve2d, 4, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_ARG_INFO(0, mode)
ZEND_ARG_INFO(0, boundary)
ZEND_ARG_INFO(0, fill_value)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, convolve2d) {
    NDArray *rtn;
    zval *a, *b;
    char *mode, *boundary;
    double fill_value = 0.0f;
    size_t size = 1;
    int imode = 0, iboundary = 0;
    ZEND_PARSE_PARAMETERS_START(4, 5)
    Z_PARAM_ZVAL(a)
    Z_PARAM_ZVAL(b)
    Z_PARAM_STRING(mode, size)
    Z_PARAM_STRING(boundary, size)
    Z_PARAM_OPTIONAL
    Z_PARAM_DOUBLE(fill_value)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL || ndb == NULL) {
        return;
    }
    switch(mode[0]) {
        case 'v':
            imode = VALID;
            break;
        case 's':
            imode = SAME;
            break;
        case 'f':
            imode = FULL;
            break;
    }
    switch(boundary[0]) {
        case 'f':
            iboundary = PAD;
            break;
        case 'w':
            iboundary = CIRCULAR;
            break;
        case 's':
            iboundary = REFLECT;
            break;
    }
    rtn = NDArray_Correlate2D(nda, ndb, imode, iboundary, NULL, 1);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::correlate2d
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_correlate2d, 4, 0, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
    ZEND_ARG_INFO(0, mode)
    ZEND_ARG_INFO(0, boundary)
    ZEND_ARG_INFO(0, fill_value)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, correlate2d) {
    NDArray *rtn;
    zval *a, *b;
    char *mode, *boundary;
    double fill_value = 0.0f;
    size_t size = 1;
    int imode = 0, iboundary = 0;
    ZEND_PARSE_PARAMETERS_START(4, 5)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
        Z_PARAM_STRING(mode, size)
        Z_PARAM_STRING(boundary, size)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(fill_value)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL || ndb == NULL) {
        return;
    }
    switch(mode[0]) {
        case 'v':
            imode = VALID;
            break;
        case 's':
            imode = SAME;
            break;
        case 'f':
            imode = FULL;
            break;
    }
    switch(boundary[0]) {
        case 'f':
            iboundary = PAD;
            break;
        case 'w':
            iboundary = CIRCULAR;
            break;
        case 's':
            iboundary = REFLECT;
            break;
    }
    rtn = NDArray_Correlate2D(nda, ndb, imode, iboundary, NULL, 0);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::norm
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_norm, 2, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, order)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, norm) {
    NDArray *rtn;
    zval *a;
    long order = 2;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(order)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Norm(nda, (int)order);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::cond
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_cond, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, cond) {
    NDArray *rtn;
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Cond(nda);
    if (rtn == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::inv
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_inv, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, inv) {
    NDArray *rtn;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Inverse(nda);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::svd
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_svd, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, svd) {
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtns = NDArray_SVD(nda);
    if (rtns == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
}

/**
 * NumPower::det
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_det, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, det) {
    NDArray *rtn, *nda;
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Det(nda);
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::sum
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sum, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sum) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    axis_i = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        rtn = reduce(nda, &axis_i, NDArray_Add_Float);
    } else {
        double value = NDArray_Sum_Float(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::min
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_min, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, min) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    double value;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        axis_i = (int)axis;
        rtn = single_reduce(nda, &axis_i, NDArray_Min);
    } else {
        value = NDArray_Min(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NumPower::max
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_max, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, max) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    double value;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        axis_i = (int)axis;
        rtn = NDArray_MaxAxis(nda, axis_i);
    } else {
        value = NDArray_Max(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_prod, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, prod) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    float value;
    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    axis_i = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        rtn = reduce(nda, &axis_i, NDArray_Multiply_Float);
    } else {
        value = NDArray_Float_Prod(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_ndarray_array, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, array) {
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    RETURN_NDARRAY(nda, return_value);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_slice, 0, 0, IS_MIXED, 0)
ZEND_ARG_VARIADIC_TYPE_INFO(0, arg, IS_MIXED, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, slice) {
    int j;
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    NDArray *rtn = NULL;
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArray** indices_axis;

    int num_args = ZEND_NUM_ARGS();

    // Check if at least one argument is passed
    if (num_args < 1) {
        php_error_docref(NULL, E_ERROR, "At least one argument is required");
        RETURN_NULL();
    }

    // Process the arguments
    zval *arg;
    int num_inputed_args = 0;
    ZEND_PARSE_PARAMETERS_START(1, -1)
    Z_PARAM_VARIADIC('+', arg, num_inputed_args)
    ZEND_PARSE_PARAMETERS_END();

    indices_axis = emalloc(sizeof(NDArray*) * num_inputed_args);
    // Access individual arguments
    for (j = 0; j < num_inputed_args; j++) {
        zval *current_arg = &arg[j];
        // Process each argument as needed
        // ...
        indices_axis[j] = ZVAL_TO_NDARRAY(current_arg);
    }
    rtn = NDArray_Slice(ndarray, indices_axis, num_inputed_args);

    for (j = 0; j < num_inputed_args; j++) {
        NDArray_FREE(indices_axis[j]);
    }
    efree(indices_axis);
    if (rtn == NULL) {
        return;
    }
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_size, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, size) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    RETURN_LONG(NDArray_NUMELEMENTS(ndarray));
}

/**
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_count, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, count) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    RETURN_LONG(NDArray_SHAPE(ndarray)[0]);
}

PHP_METHOD(NDArray, current) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArray* result  = NDArrayIteratorPHP_GET(ndarray);
    add_to_buffer(result);
    RETURN_NDARRAY(result, return_value);
}

PHP_METHOD(NDArray, key) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    RETURN_LONG(ndarray->php_iterator->current_index);
}

PHP_METHOD(NDArray, next) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArrayIteratorPHP_NEXT(ndarray);
}

PHP_METHOD(NDArray, rewind) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArrayIteratorPHP_REWIND(ndarray);
}

PHP_METHOD(NDArray, offsetExists) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    long offset;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(offset)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    if (offset < 0) {
        RETURN_BOOL(0);
        return;
    }
    if (offset > NDArray_SHAPE(ndarray)[0] - 1) {
        RETURN_BOOL(0);
        return;
    }
    RETURN_BOOL(1);
}

PHP_METHOD(NDArray, offsetGet) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *offset;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(offset)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    if (Z_TYPE_P(offset) == IS_LONG) {
        if (Z_LVAL_P(offset) < 0) {
            zend_throw_error(NULL, "Negative indexes are not implemented.");
            return;
        }

        if (Z_LVAL_P(offset) > NDArray_SHAPE(ndarray)[0] - 1) {
            zend_throw_error(NULL, "Index out of bounds");
            return;
        }
        ndarray->iterator->current_index = (int) Z_LVAL_P(offset);
        NDArray *rtn = NDArrayIterator_GET(ndarray);
        NDArrayIterator_REWIND(ndarray);
        RETURN_NDARRAY(rtn, return_value);
        return;
    }
    zend_throw_error(NULL, "Invalid offset");
    return;
}

PHP_METHOD(NDArray, offsetSet) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *offset;
    zval *value;
    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_ZVAL(offset)
    Z_PARAM_ZVAL(value)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    if (Z_TYPE_P(offset) == IS_LONG || Z_TYPE_P(offset) == IS_DOUBLE) {
        if (zval_get_long(offset) < 0) {
            zend_throw_error(NULL, "Negative indexes are not implemented.");
            return;
        }
        if (zval_get_long(offset) > NDArray_SHAPE(ndarray)[0] - 1) {
            zend_throw_error(NULL, "Index out of bounds");
            return;
        }
        NDArray* nd_value = ZVAL_TO_NDARRAY(value);
        ndarray->iterator->current_index = (int)zval_get_long(offset);
        NDArray *rtn = NDArrayIterator_GET(ndarray);
        NDArrayIterator_REWIND(ndarray);
        NDArray_Overwrite(rtn, nd_value);
        NDArray_FREE(rtn);
        CHECK_INPUT_AND_FREE(value, nd_value);
    }
    if (Z_TYPE_P(offset) == IS_OBJECT && (Z_TYPE_P(value) == IS_LONG || Z_TYPE_P(value) == IS_OBJECT)) {

    }
}

PHP_METHOD(NDArray, __serialize) {
    zval rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray must be on CPU RAM before it can be converted to a PHP array.");
        return;
    }
    if (NDArray_NDIM(array) == 0) {
        RETURN_DOUBLE(NDArray_FDATA(array)[0]);
        NDArray_FREE(array);
        return;
    }
    rtn = NDArray_ToPHPArray(array);
    RETURN_ZVAL(&rtn, 0, 0);
}

PHP_METHOD(NDArray, __unserialize) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    long offset;
    zval *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(data)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(data);
    add_to_buffer(nda);
    ZVAL_LONG(OBJ_PROP_NUM(obj, 0), NDArray_UUID(nda));
}


PHP_METHOD(NDArray, offsetUnset) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    long offset;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(offset)
    ZEND_PARSE_PARAMETERS_END();
    zend_throw_error(NULL, "Cannot unset values of NDArrays");
}

PHP_METHOD(NDArray, valid) {
    int is_valid = 0, is_done = 0;
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    is_done = NDArrayIteratorPHP_ISDONE(ndarray);
    if (is_done == 0) {
        RETURN_BOOL(1);
    }
    RETURN_BOOL(0);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ndarray_prod___toString, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, __toString) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    char *result = NDArray_Print(ndarray, 1);
    RETVAL_STRING(result);
    efree(result);
}
static const zend_function_entry class_arithmetic_methods[] = {
    ZEND_ME(ArithmeticOperand, __construct, arginfo_ArithmeticOperand_construct, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry class_NumPower_methods[] = {
    ZEND_ME(NumPower, __construct, arginfo_NumPower_construct, ZEND_ACC_PUBLIC)

    // EXTREMA
    ZEND_ME(NumPower, min, arginfo_ndarray_min, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, max, arginfo_ndarray_max, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, maximum, arginfo_ndarray_maximum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, minimum, arginfo_ndarray_minimum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, argmax, arginfo_ndarray_argmax, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, argmin, arginfo_ndarray_argmin, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // MANIPULATION
    ZEND_ME(NumPower, copy, arginfo_ndarray_copy, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, flatten, arginfo_ndarray_flat, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, atleast1d, arginfo_ndarray_atleast_1d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, atleast2d, arginfo_ndarray_atleast_2d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, atleast3d, arginfo_ndarray_atleast_3d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, transpose, arginfo_ndarray_transpose, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, append, arginfo_ndarray_append, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, expandDims, arginfo_ndarray_expand_dims, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, squeeze, arginfo_ndarray_squeeze, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, flip, arginfo_ndarray_flip, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, swapAxes, arginfo_ndarray_swapaxes, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, rollAxis, arginfo_ndarray_rollaxis, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, moveAxis, arginfo_ndarray_moveaxis, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, verticalStack, arginfo_ndarray_vstack, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, horizontalStack, arginfo_ndarray_hstack, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, depthStack, arginfo_ndarray_dstack, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, columnStack, arginfo_ndarray_column_stack, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, concatenate, arginfo_ndarray_concatenate, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // INDEXING
    ZEND_ME(NumPower, diagonal, arginfo_ndarray_diagonal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // INITIALIZERS
    ZEND_ME(NumPower, zeros, arginfo_ndarray_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, ones, arginfo_ndarray_ones, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arange, arginfo_ndarray_arange, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, identity, arginfo_ndarray_identity, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, diag, arginfo_ndarray_diag, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, full, arginfo_ndarray_full, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, array, arginfo_ndarray_array, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, fromImage, arginfo_ndarray_fromimage, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // RANDOM
    ZEND_ME(NumPower, normal, arginfo_ndarray_normal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, truncatedNormal, arginfo_ndarray_truncated_normal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, standardNormal, arginfo_ndarray_standard_normal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, poisson, arginfo_ndarray_poisson, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, uniform, arginfo_ndarray_uniform, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, randomBinomial, arginfo_ndarray_binomial, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // LINALG
    ZEND_ME(NumPower, matmul, arginfo_ndarray_matmul, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, svd, arginfo_ndarray_svd, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, det, arginfo_ndarray_det, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, dot, arginfo_ndarray_dot, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, inner, arginfo_ndarray_inner, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, outer, arginfo_ndarray_outer, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, cholesky, arginfo_ndarray_cholesky, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, qr, arginfo_ndarray_qr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, eig, arginfo_ndarray_eig, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, cond, arginfo_ndarray_cond, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, norm, arginfo_ndarray_norm, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, trace, arginfo_ndarray_trace, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, solve, arginfo_ndarray_solve, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, inv, arginfo_ndarray_inv, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, lstsq, arginfo_ndarray_lstsq, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, lu, arginfo_ndarray_lu, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, matrixRank, arginfo_ndarray_matrix_rank, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, convolve2d, arginfo_ndarray_convolve2d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, correlate2d, arginfo_ndarray_correlate2d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // DNN
    ZEND_ME(NumPower, dnnConv2dForward, arginfo_ndarray_dnn_conv2d_forward, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, dnnConv2dBackward, arginfo_ndarray_dnn_conv2d_backward, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, dnnConv1dForward, arginfo_ndarray_dnn_conv1d_forward, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // LOGIC
    ZEND_ME(NumPower, all, arginfo_ndarray_all, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, allClose, arginfo_ndarray_allclose, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, equal, arginfo_ndarray_equal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, greater, arginfo_ndarray_greater, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, greaterEqual, arginfo_ndarray_greaterequal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, less, arginfo_ndarray_less, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, lessEqual, arginfo_ndarray_lessequal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, notEqual, arginfo_ndarray_notequal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // MATH
    ZEND_ME(NumPower, abs, arginfo_ndarray_abs, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, square, arginfo_ndarray_square, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, sqrt, arginfo_ndarray_sqrt, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, exp, arginfo_ndarray_exp, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, expm1, arginfo_ndarray_expm1, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, exp2, arginfo_ndarray_exp2, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, log, arginfo_ndarray_log, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, log2, arginfo_ndarray_log2, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, logb, arginfo_ndarray_logb, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, log10, arginfo_ndarray_log10, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, log1p, arginfo_ndarray_log1p, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, sin, arginfo_ndarray_sin, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, cos, arginfo_ndarray_cos, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, tan, arginfo_ndarray_tan, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arcsin, arginfo_ndarray_arcsin, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arccos, arginfo_ndarray_arccos, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arctan, arginfo_ndarray_arctan, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arctan2, arginfo_ndarray_arctan2, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, degrees, arginfo_ndarray_degrees, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, radians, arginfo_ndarray_radians, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, sinh, arginfo_ndarray_sinh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, cosh, arginfo_ndarray_cosh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, tanh, arginfo_ndarray_tanh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arcsinh, arginfo_ndarray_arcsinh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arccosh, arginfo_ndarray_arccosh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, arctanh, arginfo_ndarray_arctanh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, rint, arginfo_ndarray_rint, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, fix, arginfo_ndarray_fix, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, floor, arginfo_ndarray_floor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, ceil, arginfo_ndarray_ceil, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, trunc, arginfo_ndarray_trunc, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, sinc, arginfo_ndarray_sinc, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, negative, arginfo_ndarray_negative, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, positive, arginfo_ndarray_positive, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, sign, arginfo_ndarray_sign, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, clip, arginfo_ndarray_clip, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, round, arginfo_ndarray_round, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, rsqrt, arginfo_ndarray_rsqrt, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, reciprocal, arginfo_ndarray_reciprocal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // STATISTICS
    ZEND_ME(NumPower, mean, arginfo_ndarray_mean, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, median, arginfo_ndarray_median, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, variance, arginfo_ndarray_variance, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, average, arginfo_ndarray_average, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, std, arginfo_ndarray_std, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, quantile, arginfo_ndarray_quantile, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // ARITHMETICS
    ZEND_ME(NumPower, add, arginfo_ndarray_add, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, subtract, arginfo_ndarray_subtract, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, pow, arginfo_ndarray_pow, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, divide, arginfo_ndarray_divide, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, multiply, arginfo_ndarray_multiply, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, sum, arginfo_ndarray_sum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, prod, arginfo_ndarray_prod, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, mod, arginfo_ndarray_mod, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, dumpDevices, arginfo_dump_devices, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, setDevice, arginfo_setdevice, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, load, arginfo_load, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, save, arginfo_save, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_FE_END
};

static const zend_function_entry class_NDArray_methods[] = {
    ZEND_ME(NDArray, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, dump, arginfo_dump, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, gpu, arginfo_gpu, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, cpu, arginfo_cpu, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, isGPU, arginfo_is_gpu, ZEND_ACC_PUBLIC)

    ZEND_ME(NDArray, size, arginfo_size, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, count, arginfo_count, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, current, arginfo_current, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, key, arginfo_key, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, next, arginfo_next, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, rewind, arginfo_rewind, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, valid, arginfo_valid, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, __toString, arginfo_ndarray_prod___toString, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, offsetExists, arginfo_offsetexists, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, offsetGet, arginfo_offsetget, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, offsetSet, arginfo_offsetset, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, offsetUnset, arginfo_offsetunset, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, __serialize, arginfo_serialize, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, __unserialize, arginfo_unserialize, ZEND_ACC_PUBLIC)

    ZEND_ME(NDArray, reshape, arginfo_reshape, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, toArray, arginfo_toArray, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, toImage, arginfo_toImage, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, slice, arginfo_slice, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, shape, arginfo_ndarray_shape, ZEND_ACC_PUBLIC)
    ZEND_ME(NDArray, fill, arginfo_fill, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

static zend_class_entry *register_class_NDArray(zend_class_entry *class_entry_Iterator, zend_class_entry *class_entry_Countable, zend_class_entry *class_entry_ArrayAccess) {
    zend_class_entry ce, *class_entry;
    INIT_CLASS_ENTRY(ce, "NDArray", class_NDArray_methods);
    ndarray_objects_init(&ce);
    ce.create_object = ndarray_create_object;
    class_entry = zend_register_internal_class(&ce);
    zend_class_implements(class_entry, 3, class_entry_Iterator, class_entry_Countable, class_entry_ArrayAccess);

    zval property_id_default_value;
    ZVAL_UNDEF(&property_id_default_value);
    zend_string *property_id_name = zend_string_init("id", sizeof("id") - 1, 1);
    zend_declare_typed_property(class_entry, property_id_name, &property_id_default_value, ZEND_ACC_PRIVATE, NULL, (zend_type) ZEND_TYPE_INIT_MASK(MAY_BE_LONG));
    zend_string_release(property_id_name);

    return class_entry;
}

static zend_class_entry *register_class_NumPower(zend_class_entry *class_entry_Iterator, zend_class_entry *class_entry_Countable, zend_class_entry *class_entry_ArrayAccess) {
    zend_class_entry ce, *class_entry;
    INIT_CLASS_ENTRY(ce, "NumPower", class_NumPower_methods);
    ndarray_objects_init(&ce);
    ce.create_object = ndarray_create_object;
    class_entry = zend_register_internal_class(&ce);
    zend_class_implements(class_entry, 3, class_entry_Iterator, class_entry_Countable, class_entry_ArrayAccess);

    zval property_id_default_value;
    ZVAL_UNDEF(&property_id_default_value);
    zend_string *property_id_name = zend_string_init("id", sizeof("id") - 1, 1);
    zend_declare_typed_property(class_entry, property_id_name, &property_id_default_value, ZEND_ACC_PRIVATE, NULL, (zend_type) ZEND_TYPE_INIT_MASK(MAY_BE_LONG));
    zend_string_release(property_id_name);

    return class_entry;
}

static zend_class_entry *register_class_ArithmeticOperand(zend_class_entry *class_entry_Iterator, zend_class_entry *class_entry_Countable, zend_class_entry *class_entry_ArrayAccess) {
    zend_class_entry ce, *class_entry;
    INIT_CLASS_ENTRY(ce, "ArithmeticOperand", class_arithmetic_methods);
    arithmetic_objects_init(&ce);
    ce.create_object = arithmetic_create_object;
    class_entry = zend_register_internal_class(&ce);
    return class_entry;
}

/**
 * MINIT
 */
PHP_MINIT_FUNCTION(ndarray) {
    phpsci_ce_NDArray = register_class_NDArray(zend_ce_iterator, zend_ce_countable, zend_ce_arrayaccess);
    phpsci_ce_ArithmeticOperand = register_class_ArithmeticOperand(zend_ce_iterator, zend_ce_countable, zend_ce_arrayaccess);
    phpsci_ce_NumPower = register_class_NumPower(zend_ce_iterator, zend_ce_countable, zend_ce_arrayaccess);
    REGISTER_LONG_CONSTANT("NUMPOWER_CPU", 0, CONST_CS | CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("NUMPOWER_CUDA", 1, CONST_CS | CONST_PERSISTENT);
    return SUCCESS;
}

PHP_RINIT_FUNCTION(ndarray) {
    unsigned int seed = time(NULL) ^ getpid() ^ clock();
    srand(seed);
    bypass_printr();
    buffer_init(2);
#if defined(ZTS) && defined(COMPILE_DL_NDARRAY)
    ZEND_TSRMLS_CACHE_UPDATE();
#endif
    return SUCCESS;
}

PHP_MINFO_FUNCTION(ndarray) {
    php_info_print_table_start();
    php_info_print_table_header(2, "support", "enabled");
    php_info_print_table_end();
}

PHP_MSHUTDOWN_FUNCTION(ndarray) {
    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ndarray) {
    char *envvar = "NDARRAY_BUFFERLEAK";
    char *envvar_vcheck = "NDARRAY_VCHECK";
    if(!getenv(envvar)) {
        buffer_free();
    }
#ifdef HAVE_CUBLAS
    if(getenv(envvar_vcheck)) {
        vmemcheck();
    }
#endif
    return SUCCESS;
}

zend_module_entry ndarray_module_entry = {
    STANDARD_MODULE_HEADER,
    "RubixNumPower",					    /* Extension name */
    ext_functions,					/* zend_function_entry */
    PHP_MINIT(ndarray),             /* PHP_MINIT - Module initialization */
    PHP_MSHUTDOWN(ndarray),							/* PHP_MSHUTDOWN - Module shutdown */
    PHP_RINIT(ndarray),			    /* PHP_RINIT - Request initialization */
    PHP_RSHUTDOWN(ndarray), /* PHP_RSHUTDOWN - Request shutdown */
    PHP_MINFO(ndarray),			    /* PHP_MINFO - Module info */
    PHP_NDARRAY_VERSION,		    /* Version */
    STANDARD_MODULE_PROPERTIES
};
/* }}} */

#ifdef COMPILE_DL_NDARRAY
# ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
# endif
ZEND_GET_MODULE(ndarray)
#endif
