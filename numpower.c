// zend_ce_iterator, zend_ce_countable, zend_ce_arrayaccess
#include <Zend/zend_interfaces.h>

// PHP_METHOD, ZEND_PARSE_PARAMETERS_START, ZEND_PARSE_PARAMETERS_END, ZEND_PARSE_PARAMETERS_NONE
#include "php.h"

// php_info_print_table_start, php_info_print_table_header, php_info_print_table_end
#include "ext/standard/info.h"

// zend_function_entry
#include "numpower_arginfo.h"

// NDArrayFactory_CreateFromZval, Create_NDArray_FromZval, NDArray_CreateFromLongScalar, NDArray_CreateFromDoubleScalar
// NDArray_Zeros,                 NDArray_FillFloat,       NDArray_Identity,             NDArray_Normal,
// NDArray_TruncatedNormal,       NDArray_Binomial,        NDArray_Poisson,              NDArray_Uniform,
// NDArray_Diag,                  NDArray_Full,            NDArray_Ones,                 NDArray_Arange,
// NDArray_Copy,
#include "src/initializers.h"

// add_to_buffer, buffer_get, buffer_ndarray_free, buffer_init, buffer_free
#include "src/buffer.h"

// NDArrayIteratorPHP_GET, NDArrayIteratorPHP_NEXT, NDArrayIteratorPHP_REWIND, NDArrayIteratorPHP_ISDONE
#include "src/iterators.h"

// phpsci_ce_NDArray, phpsci_ce_NumPower, phpsci_ce_ArithmeticOperand
#include "php_numpower.h"

// NDArray_Dump, NDArray_DumpDevices
#include "src/debug.h"

// NDArray_Add_Double,   NDArray_Add_Float,    NDArray_Subtract_Float, NDArray_Multiply_Float,
// NDArray_Divide_Float, NDArray_Pow_Float,    NDArray_Mod_Float,      NDArray_Abs,
// NDArray_Sum_Float,    NDArray_Median_Float, NDArray_Float_Prod
#include "src/ndmath/arithmetics.h"

// NDArray_ArrayEqual, NDArray_Equal,    NDArray_Greater, NDArray_GreaterEqual, NDArray_Less
// NDArray_LessEqual,  NDArray_NotEqual, NDArray_All,     NDArray_AllClose
#include "src/logic.h"

// NDArray_Reshape,     NDArray_Transpose,       NDArray_AtLeast1D,   NDArray_AtLeast2D, NDArray_AtLeast3D
// NDArray_Flatten,     NDArray_ExpandDim,       NDArray_Squeeze,     NDArray_Flip,      NDArray_SwapAxes,
// NDArray_Rollaxis,    ndarray_moveaxis,        NDArray_VSTACK,      NDArray_HSTACK,    NDArray_DSTACK,
// NDArray_ColumnStack, NDArray_ConcatenateFlat, NDArray_Concatenate, NDArray_Slice
#include "src/manipulation.h"

// Live exports of double_math.h: float_abs, float_sqrt, float_round
// (precision arg, legacy), float_arctan2 (binary, legacy).
// Every other float_* scalar helper (sin/cos/.../floor/ceil + exp/log
// family + sinc + negate/positive/sign/clip/reciprocal/rsqrt) was
// retired by the typed-unary dispatcher in src/ndmath/arithmetics.c.
#include "src/ndmath/double_math.h"

// NDArray_Matmul, NDArray_Inner,      NDArray_Outer, NDArray_Dot,   NDArray_Trace,
// NDArray_Eig,    NDArray_Cholesky,   NDArray_Solve, NDArray_Lstsq, NDArray_Qr,
// NDArray_LU,     NDArray_MatrixRank, NDArray_Norm,  NDArray_Cond,  NDArray_Inverse,
// NDArray_SVD,    NDArray_Det
#include "src/ndmath/linalg.h"

// COMPILE_DL_NDARRAY, HAVE_CUBLAS, HAVE_GD
#ifndef _MSC_VER
#include "config.h"
#endif

// NDARRAY_TYPE_FLOAT64, NDARRAY_TYPE_FLOAT32
#include "src/types.h"

// NDArray_Diagonal
#include "src/indexing.h"

// NDArray_Std, NDArray_Quantile, NDArray_Average, NDArray_Variance
#include "src/ndmath/statistics.h"

// VALID, SAME, FULL, PAD, CIRCULAR,
// REFLECT
#include "src/ndmath/signal.h"

// NDArray_ArgMinMaxCommon
#include "src/ndmath/calculation.h"

// NDArrayDNN_Conv2D_Forward, NDArray_DNN_Conv1D, NDArrayDNN_Conv2D_Backward
#include "src/dnn.h"

// zval_parameter_to_normalized_axis_argument
#include "src/sanitizers.h"

// NDArrayFactory_CreateFromZval
#include "src/ndarray/frontend/ndarray_factory.h"

// NDArray_fill
#include "src/ndarray/frontend/manipulations.h"

#ifdef HAVE_CUBLAS
  // Live cuda_float_* exports: cuda_float_abs, cuda_float_sqrt,
  // cuda_float_round (precision arg, legacy), cuda_float_arctan2
  // (binary, legacy). All other cuda_float_* trig / hyperbolic /
  // angle / rounding / sinc / negate / positive / sign / clip /
  // reciprocal / rsqrt helpers were retired by the typed-unary
  // GPU dispatcher (`cuda_<op>_{f16,f32,f64,dd}` per-dtype kernels)
  // — see the transcendental section in src/ndmath/cuda/cuda_math.h.
# include "src/ndmath/cuda/cuda_math.h"

// vmemcheck
# include "src/gpu_alloc.h"
#endif

#ifdef ZTS
# include "TSRM.h"
#endif

#ifdef HAVE_GD

#endif

/* For compatibility with older PHP versions */
#ifndef ZEND_PARSE_PARAMETERS_NONE
#define ZEND_PARSE_PARAMETERS_NONE() \
	ZEND_PARSE_PARAMETERS_START(0, 0) \
	ZEND_PARSE_PARAMETERS_END()
#endif

zend_class_entry *phpsci_ce_NDArray;
zend_class_entry *phpsci_ce_NumPower;
zend_class_entry *phpsci_ce_ArithmeticOperand;

static zend_object_handlers ndarray_object_handlers;
static zend_object_handlers numpower_object_handlers;
static zend_object_handlers arithmetic_object_handlers;

/**
 * @brief Constructor for the NDArray class.
 * 
 * ```
 * __construct(array|int|float|bool|NDArray|GdImage $input, string $dataType = "float32"): NDArray
 * ```
 * 
 * @param input    The input data to create the NDArray.
 * @param dataType The data type of the NDArray. Default is "float32".
 *                 Available values is "float32" and "double64".
 */
ZEND_BEGIN_ARG_INFO(arginfo_construct, 1)
    ZEND_ARG_INFO(0, input)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dataType, IS_STRING, 0, "float32")
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, __construct) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *input;

    char *dataType;
    size_t dataTypeLen;
    const char *ndarrayDataType;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(input)
    Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() < 2) {
        dataType = "float32";
        dataTypeLen = sizeof("float32") - 1;
    }

    if      (!strcmp(dataType, "float4"))   ndarrayDataType = NDARRAY_TYPE_FLOAT4;
    else if (!strcmp(dataType, "float8"))   ndarrayDataType = NDARRAY_TYPE_FLOAT8;
    else if (!strcmp(dataType, "float16"))  ndarrayDataType = NDARRAY_TYPE_FLOAT16;
    else if (!strcmp(dataType, "float32"))  ndarrayDataType = NDARRAY_TYPE_FLOAT32;
    else if (!strcmp(dataType, "float64"))  ndarrayDataType = NDARRAY_TYPE_FLOAT64;
    else if (!strcmp(dataType, "float128")) ndarrayDataType = NDARRAY_TYPE_FLOAT128;
    else if (!strcmp(dataType, "int8"))     ndarrayDataType = NDARRAY_TYPE_INT8;
    else if (!strcmp(dataType, "uint8"))    ndarrayDataType = NDARRAY_TYPE_UINT8;
    else if (!strcmp(dataType, "int16"))    ndarrayDataType = NDARRAY_TYPE_INT16;
    else if (!strcmp(dataType, "uint16"))   ndarrayDataType = NDARRAY_TYPE_UINT16;
    else if (!strcmp(dataType, "int32"))    ndarrayDataType = NDARRAY_TYPE_INT32;
    else if (!strcmp(dataType, "uint32"))   ndarrayDataType = NDARRAY_TYPE_UINT32;
    else if (!strcmp(dataType, "int64"))    ndarrayDataType = NDARRAY_TYPE_INT64;
    else if (!strcmp(dataType, "uint64"))   ndarrayDataType = NDARRAY_TYPE_UINT64;
    else {
        zend_throw_error(NULL,
            "Invalid data type '%s'. Supported: float4, float8, float16, float32, float64, "
            "float128, int8, uint8, int16, uint16, int32, uint32, int64, uint64", dataType);
        return;
    }

    NDArray* array = NDArrayFactory_createFromZval(input, ndarrayDataType);

    ZVAL_LONG(OBJ_PROP_NUM(obj, 0), NDArray_UUID(array));
}

/**
 * @brief Initializes a new PHP object representing an NDArray or returns a scalar value.
 *
 * This function handles conversion of an internal NDArray structure into either:
 * - A PHP object of type `phpsci_ce_NDArray` for multi-dimensional arrays
 * - A direct PHP scalar value for 0-dimensional arrays (single values)
 *
 * @param[in]  array         Pointer to the internal NDArray structure to convert. If NULL,
 *                           throws an exception and returns immediately.
 * @param[out] return_value  zval that will be initialized either as:
 *                            - An NDArray object (for dim > 0)
 *                            - A double scalar value (for 0-dim arrays)
 *
 * @note For NDArrays (dim > 0):
 *       1. Adds the array to the global buffer for tracking
 *       2. Creates a new PHP object of class `phpsci_ce_NDArray`
 *       3. Stores the array's UUID as object property #0
 *
 * @note For 0-dimensional arrays:
 *       1. Extracts the scalar value as double
 *       2. Frees the NDArray memory
 *       3. Returns the scalar directly
 *
 * @warning If array is NULL, throws an exception via RETURN_THROWS()
 * @warning The NDArray memory is managed differently based on dimensionality:
 *          - For dim > 0: Memory is tracked via the global buffer
 *          - For dim 0: Memory is freed immediately
 */
void ndarray_init_new_object(NDArray* array, zval* return_value) {
    if (array == NULL) {
        RETURN_THROWS();
        return;
    }
    if (NDArray_NDIM(array) > 0) {
        add_to_buffer(array);
        object_init_ex(return_value, phpsci_ce_NDArray);
        ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(return_value), 0), NDArray_UUID(array));
    } else {
        NDArray_ScalarToZval(array, return_value);
        NDArray_FREE(array);
    }
}

/**
 * @brief Install @p array as a freshly-built `NDArray` PHP object in
 *        @p return_value, regardless of ndim.
 *
 * The general-purpose `ndarray_init_new_object()` collapses 0-D results
 * into a primitive scalar zval (`int`/`float`/`string` per the dtype)
 * for ergonomic reasons — `NumPower::array(5.0)` returns a PHP float,
 * not an `NDArray(0-D)`. Device-transfer methods (`->gpu()` / `->cpu()`)
 * are different: callers chain them (`$a = (new NDArray(...))->gpu();`)
 * and then expect `$a` to behave like an NDArray (clone it, call other
 * methods on it). Collapsing to a primitive breaks that contract and
 * also breaks the rule that operations on a GPU-resident NDArray must
 * stay on GPU (a primitive lives entirely on the host).
 *
 * This helper always builds an NDArray object — both ndim > 0 and 0-D
 * paths produce an `NDArray` zval whose UUID property points at the
 * buffer slot.
 *
 * @param[in]  array        Freshly-built NDArray (ownership transfers to
 *                          the buffer slot).
 * @param[out] return_value zval to populate.
 */
static void ndarray_install_object(NDArray *array, zval *return_value) {
    if (array == NULL) {
        RETURN_THROWS();
        return;
    }
    add_to_buffer(array);
    object_init_ex(return_value, phpsci_ce_NDArray);
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(return_value), 0), NDArray_UUID(array));
}

ZEND_BEGIN_ARG_INFO(arginfo_gpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, gpu) {
    NDArray *rtn;
    zval *obj_zval = getThis();

    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();

#ifdef HAVE_CUBLAS
    NDArray* ndarray = NDArrayFactory_restoreFromZval(obj_zval);

    if (ndarray == NULL) {
        return;
    }

    /* Already on the GPU: no allocation, no cudaMemcpy, no buffer copy.
       Return $this unchanged regardless of ndim — see
       ndarray_install_object() for why 0-D must stay an NDArray. */
    if (NDArray_DEVICE(ndarray) == NDARRAY_DEVICE_GPU) {
        ZVAL_COPY(return_value, obj_zval);
        return;
    }

    rtn = NDArray_ToGPU(ndarray);
    ndarray_install_object(rtn, return_value);
#else
    zend_throw_error(NULL, "No GPU device available or CUDA not enabled");
    RETURN_NULL();
#endif
}

/**
 * @brief Fills the NDArray with a specified value.
 *
 * ```
 * fill(float|int|bool|string $value): void
 * ```
 *
 * String inputs preserve full precision for float128 / int64 / uint64 dtypes
 * — for those types passing a numeric string is the only way to express
 * values outside PHP's native int / float range.
 *
 * @param value The value to fill the NDArray with.
 */
ZEND_BEGIN_ARG_INFO(arginfo_fill, 1)
    ZEND_ARG_INFO(0, value)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, fill) {
    zval* value;
    zval* objZval = getThis();

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(value)
    ZEND_PARSE_PARAMETERS_END();

    if (Z_TYPE(*value) != IS_LONG && Z_TYPE(*value) != IS_DOUBLE &&
        Z_TYPE(*value) != IS_TRUE && Z_TYPE(*value) != IS_FALSE &&
        Z_TYPE(*value) != IS_STRING) {
        zend_throw_error(NULL, "Invalid value type. Supported types are: float, int, bool, string");
        return;
    }

    NDArray* ndarray = NDArrayFactory_restoreFromZval(objZval);

    if (ndarray == NULL) {
        return;
    }

    NDArray_fillByZval(ndarray, value);
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
        zend_class_entry *ce = Z_OBJCE_P(obj);
        if (instanceof_function(ce, phpsci_ce_NDArray)) {
            return buffer_get(getObjectUuid(obj));
        }
#ifdef HAVE_GD
        zend_string* class_name = Z_OBJ_P(obj)->ce->name;
        /* Check if the zend_object class name is "GdImage" */
        if (strcmp(ZSTR_VAL(class_name), "GdImage") == 0) {
            /* Legacy intake path — kept on float32/CHW/CPU so callers
               passing a GdImage where an NDArray was expected do not
               silently change their numeric type. The user-facing
               `NumPower::fromImage()` exposes the dtype / device knobs. */
            return NDArray_FromGD(obj, false, NDARRAY_TYPE_FLOAT32,
                                  NDARRAY_DEVICE_CPU);
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
                rtn[cur_index] = buffer_get(getObjectUuid(val));
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

/**
 * Compute the broadcast result shape for two NDArrays following NumPy/PyTorch
 * semantics: right-align dims, treat missing dims as 1, and when dims differ
 * the non-1 dim wins (so size-0 dims propagate, e.g. (0,) ⊕ (1,) → (0,)).
 *
 * @param a         first operand
 * @param b         second operand
 * @param out_ndim  receives the rank of the broadcast result (>= 1)
 * @return          newly emalloc'd shape array of length out_ndim, or NULL if
 *                  the shapes are not broadcast-compatible
 */
static int *ndarray_compute_broadcast_shape(const NDArray *a, const NDArray *b, int *out_ndim)
{
    int nda = NDArray_NDIM(a), ndb = NDArray_NDIM(b);
    int nd  = nda > ndb ? nda : ndb;
    /* The arithmetic call sites assume out_ndim >= 1 (they emalloc strides
       indexed by ndim) — when both operands are 0-D scalars the broadcast
       result is conceptually shape () but the rest of the code path treats
       the empty short-circuit only for arrays with a zero dim, which a 0-D
       scalar does not have, so this branch is unreachable in practice. */
    if (nd < 1) nd = 1;
    int *shape = emalloc(sizeof(int) * (size_t)nd);
    for (int k = 0; k < nd; k++) {
        int da = (k < nda) ? NDArray_SHAPE(a)[nda - 1 - k] : 1;
        int db = (k < ndb) ? NDArray_SHAPE(b)[ndb - 1 - k] : 1;
        if (da != db && da != 1 && db != 1) {
            efree(shape);
            return NULL;
        }
        shape[nd - 1 - k] = (da == 1) ? db : da;
    }
    *out_ndim = nd;
    return shape;
}

/* Forward declaration: ndarray_arith_dispatch (below) calls into the
   typed-kernel dispatcher whose body lives later in this file. Same
   translation unit so the linker has no work to do; declaring up front
   keeps the helper-then-dispatcher ordering readable. */
static NDArray *ndarray_promote_and_op(zend_uchar opcode, NDArray *nda,
                                        NDArray *ndb,
                                        const char **result_type_out);

/**
 * Apply ZEND_DIV's "true division" dtype rule: integer operands divide to
 * float (float32 for narrow ints, float64 for 32/64-bit ints). Matches
 * PyTorch and the same logic used in the non-empty arithmetic path.
 *
 * @param result_type promoted dtype before applying the division rule
 * @return            adjusted dtype to use for the division result
 */
static const char *ndarray_div_promote(const char *result_type)
{
    int is_int_result =
        (!strcmp(result_type, "int8")   || !strcmp(result_type, "uint8")  ||
         !strcmp(result_type, "int16")  || !strcmp(result_type, "uint16") ||
         !strcmp(result_type, "int32")  || !strcmp(result_type, "uint32") ||
         !strcmp(result_type, "int64")  || !strcmp(result_type, "uint64"));
    if (!is_int_result) return result_type;
    if (!strcmp(result_type, "int32") || !strcmp(result_type, "uint32") ||
        !strcmp(result_type, "int64") || !strcmp(result_type, "uint64")) {
        return "float64";
    }
    return "float32";
}

/**
 * @brief Return non-zero when @p z is a PHP scalar promotable to a 0-D
 *        NDArray for weak-scalar arithmetic.
 *
 * Mirrors the accepted-types set of `NDArray_EncodeZvalToDtype`:
 * IS_LONG, IS_DOUBLE, IS_STRING, IS_TRUE, IS_FALSE. Every other zval
 * type (arrays, objects, null, references) returns zero so the caller
 * can fall back to the generic `ZVAL_TO_NDARRAY` path.
 *
 * @param[in] z Candidate zval.
 * @return Non-zero if @p z can be promoted; zero otherwise.
 */
static int ndarray_is_promotable_scalar(zval *z)
{
    int t = Z_TYPE_P(z);
    return t == IS_LONG || t == IS_DOUBLE || t == IS_STRING
        || t == IS_TRUE || t == IS_FALSE;
}

/**
 * @brief Choose the dtype of a weak-scalar promoted alongside @p tensor_dt.
 *
 * PyTorch "weak scalar" promotion rules:
 *  - IS_LONG / IS_STRING / IS_TRUE / IS_FALSE: adopt the tensor's dtype.
 *  - IS_DOUBLE: adopt the tensor's dtype when it is already floating point;
 *    otherwise promote to `float64` so `int_tensor + 1.5` returns a float
 *    result instead of truncating the fractional part.
 *
 * @param[in] tensor_dt Canonical dtype string of the peer NDArray operand.
 * @param[in] z         PHP scalar being promoted.
 * @return Target dtype string for the scalar's 0-D NDArray.
 */
static const char *ndarray_pick_scalar_dtype(const char *tensor_dt, zval *z)
{
    int is_int_tensor =
        (!strcmp(tensor_dt, "int8")   || !strcmp(tensor_dt, "uint8")  ||
         !strcmp(tensor_dt, "int16")  || !strcmp(tensor_dt, "uint16") ||
         !strcmp(tensor_dt, "int32")  || !strcmp(tensor_dt, "uint32") ||
         !strcmp(tensor_dt, "int64")  || !strcmp(tensor_dt, "uint64"));
    if (is_int_tensor && Z_TYPE_P(z) == IS_DOUBLE) {
        return "float64";
    }
    return tensor_dt;
}

/**
 * @brief Allocate a 0-D NDArray of dtype @p target_dt and encode @p z into it.
 *
 * Routes through `NDArray_EncodeZvalToDtype` so encoding is loss-free for
 * the wide dtypes — `float128` strings flow through `strtoflt128` (or the
 * DD parser), `int64` / `uint64` through `strtoll` / `strtoull`. Every other
 * dtype coerces the scalar through `double`, which represents its full range
 * exactly. The resulting NDArray lives in CPU RAM; the upstream device
 * migration in `ndarray_promote_and_op` moves it to GPU when needed.
 *
 * @param[in] z         PHP scalar (IS_LONG / IS_DOUBLE / IS_STRING /
 *                      IS_TRUE / IS_FALSE).
 * @param[in] target_dt Canonical dtype string.
 * @return Caller-owned 0-D NDArray on success, NULL on validation error
 *         (with a PHP exception in flight).
 */
static NDArray *ndarray_make_typed_scalar(zval *z, const char *target_dt)
{
    int *shape0 = emalloc(sizeof(int));
    shape0[0] = 1;
    NDArray *r = NDArray_Empty(shape0, 0, target_dt, NDARRAY_DEVICE_CPU);
    if (r == NULL) {
        return NULL;
    }
    if (!NDArray_EncodeZvalToDtype(z, target_dt, (char *)NDArray_DATA(r))) {
        NDArray_FREE(r);
        return NULL;
    }
    return r;
}

/**
 * @brief Resolve a PHP operand zval to an NDArray, honouring weak-scalar
 *        promotion against the peer operand's dtype.
 *
 * Handles both the NDArray-array and NDArray-scalar cases used by every
 * binary arithmetic op:
 *  - When @p other is an NDArray and @p value is a promotable scalar
 *    (IS_LONG/IS_DOUBLE/IS_STRING/IS_TRUE/IS_FALSE), allocate a 0-D NDArray
 *    of @p other's dtype and encode @p value into it with full precision
 *    via `NDArray_EncodeZvalToDtype`. This is the only path that keeps
 *    `float128` / `int64` / `uint64` strings loss-free end-to-end.
 *  - Otherwise fall back to `ZVAL_TO_NDARRAY`, which already handles
 *    arrays, NDArrays, integers, and doubles. A bare string with no peer
 *    NDArray to anchor on throws a clear error here (we cannot guess a
 *    dtype).
 *
 * @param[in]  value     PHP zval to resolve.
 * @param[in]  other     Peer NDArray, or NULL when none exists.
 * @param[out] is_owned  Receives 1 when the helper allocated a fresh NDArray
 *                       that the caller must `NDArray_FREE` (rather than
 *                       releasing via `CHECK_INPUT_AND_FREE`).
 * @return NDArray operand on success, NULL on error (PHP exception in flight).
 */
static NDArray *ndarray_arith_resolve_operand(zval *value, NDArray *other,
                                              int *is_owned)
{
    *is_owned = 0;
    if (other != NULL && ndarray_is_promotable_scalar(value)) {
        const char *target = ndarray_pick_scalar_dtype(NDArray_TYPE(other), value);
        NDArray *r = ndarray_make_typed_scalar(value, target);
        if (r == NULL) {
            return NULL;
        }
        *is_owned = 1;
        return r;
    }
    if (Z_TYPE_P(value) == IS_STRING) {
        zend_throw_error(NULL,
            "Cannot infer dtype for a string scalar without an NDArray peer.");
        return NULL;
    }
    return ZVAL_TO_NDARRAY(value);
}

/**
 * @brief Single binary-arithmetic dispatcher shared by every PHP entry point.
 *
 * Both `NumPower::add/sub/mul/div/pow/mod` and the operator-overload path
 * (`$a + $b`, …) funnel through here. The function:
 *  1. Resolves each operand via `ndarray_arith_resolve_operand`, applying
 *     PyTorch's weak-scalar promotion when one side is a non-NDArray scalar.
 *     String scalars adopt the peer NDArray's dtype so `float128` / `uint64`
 *     precision is preserved end-to-end.
 *  2. Checks broadcast compatibility.
 *  3. Calls `ndarray_promote_and_op`, which routes the operands to the right
 *     typed CPU or GPU kernel.
 *  4. Releases every transient operand exactly once (whether owned or
 *     borrowed from a PHP zval).
 *  5. Installs the result via `ndarray_init_new_object`, which collapses
 *     a 0-D result to the dtype-correct PHP scalar (string for `float128`/
 *     `uint64`, int for the remaining integer dtypes, float for the
 *     remaining floats).
 *
 * @param[in]  opcode ZEND_ADD / SUB / MUL / DIV / POW / MOD.
 * @param[in]  op1    First operand zval (NDArray / array / scalar).
 * @param[in]  op2    Second operand zval.
 * @param[out] result zval populated with the arithmetic result.
 * @return SUCCESS on success, FAILURE when an operand cannot be resolved,
 *         shapes are not broadcastable, or the typed kernel itself failed.
 */
static int ndarray_arith_dispatch(zend_uchar opcode, zval *op1, zval *op2,
                                   zval *result)
{
    int op1_is_nda = (Z_TYPE_P(op1) == IS_OBJECT &&
                      instanceof_function(Z_OBJCE_P(op1), phpsci_ce_NDArray));
    int op2_is_nda = (Z_TYPE_P(op2) == IS_OBJECT &&
                      instanceof_function(Z_OBJCE_P(op2), phpsci_ce_NDArray));

    NDArray *nda = NULL, *ndb = NULL;
    int nda_owned = 0, ndb_owned = 0;

    if (op1_is_nda && !op2_is_nda) {
        nda = ZVAL_TO_NDARRAY(op1);
        if (nda == NULL) return FAILURE;
        ndb = ndarray_arith_resolve_operand(op2, nda, &ndb_owned);
        if (ndb == NULL) {
            CHECK_INPUT_AND_FREE(op1, nda);
            return FAILURE;
        }
    } else if (op2_is_nda && !op1_is_nda) {
        ndb = ZVAL_TO_NDARRAY(op2);
        if (ndb == NULL) return FAILURE;
        nda = ndarray_arith_resolve_operand(op1, ndb, &nda_owned);
        if (nda == NULL) {
            CHECK_INPUT_AND_FREE(op2, ndb);
            return FAILURE;
        }
    } else {
        /* Either both NDArrays or neither. For both-scalar (or both-array,
           or one-array-one-array) the order matters only when exactly one
           operand is `IS_STRING`: the string needs a peer NDArray to
           anchor its dtype against. Resolve the non-string side first when
           the asymmetry shows up, then anchor the string against it; this
           keeps `add('1.5', 2)` and `add(2, '1.5')` symmetrical instead of
           letting the first form throw while the second succeeds. */
        int op1_is_str = Z_TYPE_P(op1) == IS_STRING;
        int op2_is_str = Z_TYPE_P(op2) == IS_STRING;
        if (op1_is_str && !op2_is_str) {
            ndb = ndarray_arith_resolve_operand(op2, NULL, &ndb_owned);
            if (ndb == NULL) return FAILURE;
            nda = ndarray_arith_resolve_operand(op1, ndb, &nda_owned);
            if (nda == NULL) {
                if (ndb_owned) NDArray_FREE(ndb); else CHECK_INPUT_AND_FREE(op2, ndb);
                return FAILURE;
            }
        } else {
            nda = ndarray_arith_resolve_operand(op1, NULL, &nda_owned);
            if (nda == NULL) return FAILURE;
            ndb = ndarray_arith_resolve_operand(op2, nda, &ndb_owned);
            if (ndb == NULL) {
                if (nda_owned) NDArray_FREE(nda); else CHECK_INPUT_AND_FREE(op1, nda);
                return FAILURE;
            }
        }
    }

    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can't broadcast arrays with incompatible shapes.");
        if (nda_owned) NDArray_FREE(nda); else CHECK_INPUT_AND_FREE(op1, nda);
        if (ndb_owned) NDArray_FREE(ndb); else CHECK_INPUT_AND_FREE(op2, ndb);
        return FAILURE;
    }

    NDArray *rtn = ndarray_promote_and_op(opcode, nda, ndb, NULL);
    if (nda_owned) NDArray_FREE(nda); else CHECK_INPUT_AND_FREE(op1, nda);
    if (ndb_owned) NDArray_FREE(ndb); else CHECK_INPUT_AND_FREE(op2, ndb);

    if (rtn == NULL) {
        return FAILURE;
    }
    rtn->uuid = -1;
    ndarray_init_new_object(rtn, result);
    return SUCCESS;
}

static NDArray *ndarray_promote_and_op(zend_uchar opcode, NDArray *nda, NDArray *ndb,
                                        const char **result_type_out)
{
    int dev_a = NDArray_DEVICE(nda);
    int dev_b = NDArray_DEVICE(ndb);

    /* Device handling: when at least one operand is on GPU, keep computation
       on GPU. Scalar + GPU(any dtype): migrate scalar to GPU (carrying the
       array's dtype). Real-array CPU + real-array GPU still throws. */
    NDArray *nda_dev_migrated = NULL, *ndb_dev_migrated = NULL;
    if (dev_a != dev_b) {
        int a_is_scalar = (NDArray_NDIM(nda) == 0);
        int b_is_scalar = (NDArray_NDIM(ndb) == 0);
        if (!a_is_scalar && !b_is_scalar) {
            zend_throw_error(NULL,
                "Device mismatch, both NDArray MUST be in the same device.");
            return NULL;
        }
        /* Migrate scalar to the other operand's device so we stay on GPU
           for the compute. The migration uses NDArray_ToGPU / ToCPU. */
        if (a_is_scalar) {
            NDArray *moved = (dev_b == NDARRAY_DEVICE_GPU)
                ? NDArray_ToGPU(nda) : NDArray_ToCPU(nda);
            if (moved == NULL) return NULL;
            nda_dev_migrated = moved;
            nda = moved;
            dev_a = dev_b;
        } else {
            NDArray *moved = (dev_a == NDARRAY_DEVICE_GPU)
                ? NDArray_ToGPU(ndb) : NDArray_ToCPU(ndb);
            if (moved == NULL) return NULL;
            ndb_dev_migrated = moved;
            ndb = moved;
            dev_b = dev_a;
        }
    }

    int both_gpu = (dev_a == NDARRAY_DEVICE_GPU);

    /* Empty broadcast short-circuit: when either operand has any zero dim,
       the result is empty (NumPy/PyTorch semantics). The downstream kernels
       and NDArray_Broadcast itself do not handle this correctly — Broadcast
       allocates EmptyLike(non-empty) and the kernel reads uninitialized
       memory, leaking garbage into the result. Bypass the kernel entirely
       and return a typed empty NDArray with the broadcast shape. */
    if (NDArray_NUMELEMENTS(nda) == 0 || NDArray_NUMELEMENTS(ndb) == 0) {
        int rndim = 0;
        int *rshape = ndarray_compute_broadcast_shape(nda, ndb, &rndim);
        if (rshape == NULL) {
            zend_throw_error(NULL, "Can't broadcast arrays with incompatible shapes.");
            if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
            if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
            return NULL;
        }
        const char *empty_result_type = promote_dtype(NDArray_TYPE(nda), NDArray_TYPE(ndb));
        if (opcode == ZEND_DIV) {
            empty_result_type = ndarray_div_promote(empty_result_type);
        }
        if (result_type_out) *result_type_out = empty_result_type;
        NDArray *empty_rtn = NDArray_Empty(rshape, rndim, empty_result_type, dev_a);
        if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
        if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
        return empty_rtn;
    }

    if (both_gpu) {
        /* GPU stays on GPU for every supported dtype. We promote types, cast
           on GPU via NDArray_AsType (now GPU-aware), call the typed GPU binop,
           then cast back. No CPU round-trip for float32, float64, float16,
           int8..uint64, and (via dd kernels) float128.
           float4/float8 fall back to CPU because there are no native CUDA
           intrinsics and they're 1-byte values (we go through NDArray_AsType
           which already routes them through CPU for those source/target types). */
        const char *gpu_result_type = promote_dtype(NDArray_TYPE(nda), NDArray_TYPE(ndb));

        if (opcode == ZEND_DIV) {
            gpu_result_type = ndarray_div_promote(gpu_result_type);
        }
        const char *gpu_comp_type = compute_dtype_for_arithmetic(gpu_result_type);
        if (result_type_out) *result_type_out = gpu_result_type;

        NDArray *gpu_a_cast = NULL, *gpu_b_cast = NULL;
        if (!is_type(NDArray_TYPE(nda), gpu_comp_type)) {
            gpu_a_cast = NDArray_AsType(nda, gpu_comp_type);
            if (gpu_a_cast == NULL) {
                if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
                if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
                return NULL;
            }
        }
        if (!is_type(NDArray_TYPE(ndb), gpu_comp_type)) {
            gpu_b_cast = NDArray_AsType(ndb, gpu_comp_type);
            if (gpu_b_cast == NULL) {
                if (gpu_a_cast) NDArray_FREE(gpu_a_cast);
                if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
                if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
                return NULL;
            }
        }
        NDArray *ga = gpu_a_cast ? gpu_a_cast : nda;
        NDArray *gb = gpu_b_cast ? gpu_b_cast : ndb;
        /* If AsType pulled to CPU (float4/float8/float128 with no GPU cast
           kernel), continue with CPU compute below by falling through. */
        if (NDArray_DEVICE(ga) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(gb) == NDARRAY_DEVICE_GPU) {
            NDArray *gr = NDArray_TypedBinOp_GPU(opcode, ga, gb);
            if (gpu_a_cast) NDArray_FREE(gpu_a_cast);
            if (gpu_b_cast) NDArray_FREE(gpu_b_cast);
            if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
            if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
            if (gr == NULL) return NULL;
            /* Cast back to result_type if different from comp_type. */
            if (!is_type(gpu_comp_type, gpu_result_type)) {
                NDArray *gr_cast = NDArray_AsType(gr, gpu_result_type);
                NDArray_FREE(gr);
                gr = gr_cast;
            }
            return gr;
        }
        /* Fell through: at least one cast went CPU-side (float4/8/128 source
           or destination). Continue with CPU compute by treating both as CPU
           operands. Free any GPU temporaries created above. */
        if (gpu_a_cast && NDArray_DEVICE(gpu_a_cast) == NDARRAY_DEVICE_CPU) {
            nda = gpu_a_cast;
        } else if (gpu_a_cast) {
            /* gpu_a_cast is on GPU but other side went CPU — pull this down. */
            NDArray *cpu_a = NDArray_ToCPU(gpu_a_cast);
            NDArray_FREE(gpu_a_cast);
            gpu_a_cast = cpu_a;
            nda = gpu_a_cast;
        } else if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            NDArray *cpu_a = NDArray_ToCPU(nda);
            if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
            nda_dev_migrated = cpu_a;
            nda = cpu_a;
        }
        if (gpu_b_cast && NDArray_DEVICE(gpu_b_cast) == NDARRAY_DEVICE_CPU) {
            ndb = gpu_b_cast;
        } else if (gpu_b_cast) {
            NDArray *cpu_b = NDArray_ToCPU(gpu_b_cast);
            NDArray_FREE(gpu_b_cast);
            gpu_b_cast = cpu_b;
            ndb = gpu_b_cast;
        } else if (NDArray_DEVICE(ndb) == NDARRAY_DEVICE_GPU) {
            NDArray *cpu_b = NDArray_ToCPU(ndb);
            if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
            ndb_dev_migrated = cpu_b;
            ndb = cpu_b;
        }
        /* Reassign migrations to be cleaned up at the end. */
        if (gpu_a_cast && !nda_dev_migrated) nda_dev_migrated = gpu_a_cast;
        else if (gpu_a_cast)                 NDArray_FREE(gpu_a_cast);
        if (gpu_b_cast && !ndb_dev_migrated) ndb_dev_migrated = gpu_b_cast;
        else if (gpu_b_cast)                 NDArray_FREE(gpu_b_cast);
        both_gpu = 0;
    }

    const char *result_type = promote_dtype(NDArray_TYPE(nda), NDArray_TYPE(ndb));

    /* PyTorch: true division ("/") always returns a float dtype, even for
       integer inputs. int32 / 2 → float32 (not int32 truncated). */
    if (opcode == ZEND_DIV) {
        result_type = ndarray_div_promote(result_type);
    }

    const char *comp_type   = compute_dtype_for_arithmetic(result_type);
    if (result_type_out) *result_type_out = result_type;

    NDArray *nda_cast = NULL, *ndb_cast = NULL;

    if (!is_type(NDArray_TYPE(nda), comp_type)) {
        nda_cast = NDArray_AsType(nda, comp_type);
        if (nda_cast == NULL) {
            if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
            if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
            return NULL;
        }
    }
    if (!is_type(NDArray_TYPE(ndb), comp_type)) {
        ndb_cast = NDArray_AsType(ndb, comp_type);
        if (ndb_cast == NULL) {
            if (nda_cast) NDArray_FREE(nda_cast);
            if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
            if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);
            return NULL;
        }
    }

    NDArray *a = nda_cast ? nda_cast : nda;
    NDArray *b = ndb_cast ? ndb_cast : ndb;
    int use_double   = is_type(comp_type, NDARRAY_TYPE_FLOAT64);
    int use_float128 = is_type(comp_type, NDARRAY_TYPE_FLOAT128);
    int use_int_native =
        is_type(comp_type, "int8")  || is_type(comp_type, "uint8")  ||
        is_type(comp_type, "int16") || is_type(comp_type, "uint16") ||
        is_type(comp_type, "int32") || is_type(comp_type, "uint32") ||
        is_type(comp_type, "int64") || is_type(comp_type, "uint64");

    NDArray *rtn = NULL;
    if (use_int_native) {
        /* Native integer CPU kernel — keeps every integer dtype native so
           PyTorch's modular wrap-around survives end-to-end. The legacy
           float64 round-trip silently rounded `int32 * int32` past 2^53
           (e.g. `(2^28+1)^2` returned 536870912 instead of 536870913)
           and the matching `cuda_cast_f64_to_i32` saturated rather than
           wrapping, so CPU and GPU diverged on the same input. */
        rtn = NDArray_TypedBinOp_CPU_Int(opcode, a, b);
    } else {
        switch (opcode) {
        case ZEND_ADD:
            rtn = use_float128 ? NDArray_Add_Float128(a, b)
                : use_double   ? NDArray_Add_Double(a, b)
                               : NDArray_Add_Float(a, b);      break;
        case ZEND_SUB:
            rtn = use_float128 ? NDArray_Subtract_Float128(a, b)
                : use_double   ? NDArray_Subtract_Double(a, b)
                               : NDArray_Subtract_Float(a, b); break;
        case ZEND_MUL:
            rtn = use_float128 ? NDArray_Multiply_Float128(a, b)
                : use_double   ? NDArray_Multiply_Double(a, b)
                               : NDArray_Multiply_Float(a, b); break;
        case ZEND_DIV:
            rtn = use_float128 ? NDArray_Divide_Float128(a, b)
                : use_double   ? NDArray_Divide_Double(a, b)
                               : NDArray_Divide_Float(a, b);   break;
        case ZEND_POW:
            rtn = use_float128 ? NDArray_Pow_Float128(a, b)
                : use_double   ? NDArray_Pow_Double(a, b)
                               : NDArray_Pow_Float(a, b);      break;
        case ZEND_MOD:
            rtn = use_float128 ? NDArray_Mod_Float128(a, b)
                : use_double   ? NDArray_Mod_Double(a, b)
                               : NDArray_Mod_Float(a, b);      break;
        default:
            break;
        }
    }

    if (nda_cast) NDArray_FREE(nda_cast);
    if (ndb_cast) NDArray_FREE(ndb_cast);
    if (nda_dev_migrated) NDArray_FREE(nda_dev_migrated);
    if (ndb_dev_migrated) NDArray_FREE(ndb_dev_migrated);

    /* Cast back to result_type if computation used a wider type (e.g. float16→float32→float16) */
    if (rtn != NULL && !is_type(comp_type, result_type)) {
        NDArray *rtn_cast = NDArray_AsType(rtn, result_type);
        NDArray_FREE(rtn);
        rtn = rtn_cast;
    }

    return rtn;
}

/**
 * @brief Operator-overload bridge — same path as the explicit PHP methods.
 *
 * PHP calls this through `ndarray_object_handlers.do_operation` for
 * `$a + $b`, `$a - $b`, … . Opcodes outside the supported arithmetic
 * set return FAILURE so the engine falls back to its default handling
 * (e.g. string concatenation via `__toString`). All accepted opcodes
 * go through the shared `ndarray_arith_dispatch`, which implements
 * weak-scalar promotion (including IS_STRING for fp128/int64/uint64),
 * broadcasting, typed dispatch, and 0-D scalar return.
 */
static int ndarray_do_operation_ex(zend_uchar opcode, zval *result, zval *op1, zval *op2) { /* {{{ */
    switch (opcode) {
        case ZEND_ADD: case ZEND_SUB: case ZEND_MUL:
        case ZEND_DIV: case ZEND_POW: case ZEND_MOD:
            break;
        default:
            return FAILURE;
    }
    return ndarray_arith_dispatch(opcode, op1, op2, result);
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

    if (Z_TYPE_P(obj_uuid) == IS_LONG) {
        buffer_ndarray_free(Z_LVAL_P(obj_uuid));
    }

    zend_object_std_dtor(object); // всегда вызывается
}

/**
 * @brief `clone $ndarray` handler — deep-copy the underlying buffer and
 *        attach our custom operator/iterator handlers to the new object.
 *
 * The default `zend_objects_clone_obj` allocates the destination through
 * a path that may not invoke `ndarray_create_object`, so the cloned
 * object can end up with `&std_object_handlers` instead of
 * `&ndarray_object_handlers`. That makes `$clone + 2` throw
 * `Unsupported operand types: NDArray + int` because the do_operation
 * slot is NULL on the clone. We allocate ourselves to guarantee the
 * right handler table.
 *
 * Beyond fixing the handler, we also do a real device-aware deep copy
 * via `NDArray_Copy()` (CPU buffers via `memcpy`, GPU buffers via
 * `cudaMemcpy DeviceToDevice`). Without the deep copy the clone's `id`
 * property would still point at the original's buffer slot, so a later
 * `$clone->cpu()` or `$clone->gpu()` would also move the source.
 *
 * @param[in] old_object Source NDArray PHP object.
 * @return Freshly-allocated cloned `NDArray` PHP object with a private
 *         deep copy of the buffer on the same device as the source.
 */
static zend_object *ndarray_clone_obj(zend_object *old_object) {
    NDArrayObject *intern = (NDArrayObject *) zend_object_alloc(
        sizeof(NDArrayObject), old_object->ce);
    zend_object_std_init(&intern->std, old_object->ce);
    object_properties_init(&intern->std, old_object->ce);
    intern->std.handlers = &ndarray_object_handlers;

    zval *src_id = OBJ_PROP_NUM(old_object, 0);
    if (Z_TYPE_P(src_id) == IS_LONG) {
        NDArray *src = buffer_get((int) Z_LVAL_P(src_id));
        if (src != NULL) {
            NDArray *copy = NDArray_Copy(src, NDArray_DEVICE(src));
            if (copy != NULL) {
                add_to_buffer(copy);
                ZVAL_LONG(OBJ_PROP_NUM(&intern->std, 0), NDArray_UUID(copy));
            }
        }
    }

    return &intern->std;
}

static void ndarray_objects_init(zend_class_entry *class_type) {
    memcpy(&ndarray_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    ndarray_object_handlers.compare = ndarray_objects_compare;
    ndarray_object_handlers.do_operation = ndarray_do_operation;
    ndarray_object_handlers.free_obj = ndarray_destructor;
    ndarray_object_handlers.clone_obj = ndarray_clone_obj;
}

static void numpower_objects_init(zend_class_entry *class_type) {
    memcpy(&numpower_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));
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
        return buffer_get(getObjectUuid(obj));
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

    ndarray_init_new_object(array1, &a);
    ndarray_init_new_object(array2, &b);
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

/**
 * @brief Intercept the built-in `print_r` to make NDArray-aware output
 *        available transparently.
 *
 * Locates the global `print_r` function entry, repoints its internal
 * handler at our `ZEND_FN(print_r_)` implementation, and renames the
 * entry to `print_r_` so users can still reach the original behaviour
 * through that name. Called from `PHP_RINIT_FUNCTION(ndarray)`; the
 * rename persists for the process lifetime (subsequent RINITs find
 * no `print_r` to rename and exit early).
 *
 * Ownership of the new function-name string transfers to the
 * function-table entry. The engine's `zend_internal_function_dtor`
 * runs at MSHUTDOWN and releases `function_name` exactly once — so
 * the new `zend_string` is created with `refcount = 1` and **no extra
 * `zend_string_addref` is performed**. The legacy implementation
 * called `addref` after the assignment, leaving `refcount = 2`; the
 * engine's single release at shutdown then dropped it to `1` and the
 * 40-byte string was never freed (a once-per-process leak).
 *
 * The lookup-key `functionToRename` is allocated with `persistent = 0`
 * because it lives only inside this function call; it is released at
 * the end. Releasing the *original* `print_r` `function_name` is safe
 * even with `persistent = 0` because that string is engine-interned
 * (`ZSTR_IS_INTERNED` short-circuits the release).
 */
void
bypass_printr() {
    zend_string *functionToRename = zend_string_init("print_r",
                                                       strlen("print_r"), 0);
    zend_function *functionEntry = zend_hash_find_ptr(EG(function_table),
                                                        functionToRename);
    if (functionEntry != NULL) {
        zend_string *newFunctionName = zend_string_init("print_r_",
                                                          strlen("print_r_"),
                                                          1);
        zend_string_release_ex(functionEntry->common.function_name, 0);
        functionEntry->common.function_name        = newFunctionName;
        functionEntry->internal_function.handler   = ZEND_FN(print_r_);
        /* No `zend_string_addref` here: the engine's
           `zend_internal_function_dtor` will release this string exactly
           once at MSHUTDOWN, so we hand over the single reference we
           own. The legacy extra addref was the source of the 40-byte
           once-per-process leak surfaced by valgrind. */
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
        NDArray_ScalarToZval(array, return_value);
        return;
    }

    rtn = NDArray_ToPHPArray(array);
    RETURN_ZVAL(&rtn, 0, 0);
}

/**
 * @brief `NDArray::toImage($alpha = null): \GdImage` — convert any
 *        RGB-shaped NDArray (CHW or HWC, CPU or GPU, any dtype) into a
 *        GD truecolor image.
 *
 * The legacy implementation only accepted float32 CHW arrays resident in
 * CPU RAM, forcing every GPU pipeline to do an explicit `->cpu()` and
 * every HWC pipeline (notably the default output of `fromImage`) to first
 * transpose. The shape gate also rejected HWC outright with a misleading
 * "must be 3-dimensional" message even though the array was 3-D. This
 * version detects CHW vs HWC at the C layer, stages GPU operands via
 * `NDArray_TypedD2H`, and clamps every channel to [0, 255] before bit-
 * packing so float values that drift outside the displayable range do
 * not corrupt adjacent color bytes.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval populated with a fresh `\GdImage`.
 *
 * @throws \Error When called against a build without GD; when the array is
 *                not 3-D with a channel axis of size 3; when an alpha
 *                argument is provided but its shape disagrees with the
 *                image's `(H, W)`.
 */
ZEND_BEGIN_ARG_INFO(arginfo_toImage, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, alpha, NDArray, 1, "null")
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
    if (array == NULL) {
        return;
    }
    if (alpha != NULL && Z_TYPE_P(alpha) != IS_NULL) {
        n_alpha = ZVAL_TO_NDARRAY(alpha);
        if (n_alpha == NULL) {
            return;
        }
    }
#ifdef HAVE_GD
    /* Layout / dtype / device checks all live in NDArray_ToGD so that
       the future toImage() shape detector stays single-sourced. */
    NDArray_ToGD(array, n_alpha, return_value);
    if (alpha != NULL && Z_TYPE_P(alpha) != IS_NULL) {
        CHECK_INPUT_AND_FREE(alpha, n_alpha);
    }
#else
    (void)n_alpha;
    zend_throw_error(NULL, "NDArray::toImage() requires the extension to be built with GD support.");
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
    /* Already on the CPU: no allocation, no copy. Return $this regardless
       of ndim — see ndarray_install_object() for why 0-D arrays must stay
       NDArray objects rather than collapsing to a primitive. */
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
        ZVAL_COPY(return_value, obj_zval);
        return;
    }
    rtn = NDArray_ToCPU(array);
    ndarray_install_object(rtn, return_value);
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
/**
 * @brief NumPower::dumpDevices() — list available CUDA devices.
 *
 * Print information about every available CUDA device to PHP's stdout
 * stream. When the extension is built without CUDA, prints a single line
 * stating that no GPU devices are available. Output is routed through
 * php_printf so that PHP output buffering and SAPIs other than CLI work.
 */
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
    ndarray_init_new_object(rtn, return_value);
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
/**
 * @brief NumPower::setDevice($deviceId) — switch the active CUDA device.
 *
 * Thin wrapper around `cudaSetDevice()` that adds range checking and
 * CUDA-runtime error reporting. The device id must be in
 * `[0, deviceCount)`; negative or out-of-range ids are rejected before
 * touching the runtime. When CUDA is not compiled in, the call throws
 * so that the stub's `@throws \Error` contract is honored.
 *
 * In ZTS builds the selection is per-thread (CUDA semantics), so it
 * only applies to the current PHP request's worker thread.
 *
 * @param[in] deviceId Zero-based CUDA device id (PHP `int`).
 *
 * @throws \Error If CUDA is not compiled in, the runtime is unavailable,
 *                no devices are visible, the id is out of range, or
 *                `cudaSetDevice` fails.
 */
PHP_METHOD(NumPower, setDevice) {
    zend_long deviceId;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(deviceId)
    ZEND_PARSE_PARAMETERS_END();
#ifdef HAVE_CUBLAS
    /* Three runtime states are possible on a CUDA-linked build:
       (a) the toolkit is present AND at least one GPU is visible → take
           the normal validation + cudaSetDevice path;
       (b) the toolkit is linked but cudaGetDeviceCount fails or reports
           0 devices (driver missing, GPU not present, container without
           the device passed in) → behave like a CPU-only build so the
           "No GPU device available or CUDA not enabled" contract holds
           regardless of build flavor;
       (c) cudaSetDevice itself errors → surface the runtime message. */
    int numDevices = 0;
    cudaError_t err = cudaGetDeviceCount(&numDevices);
    if (err != cudaSuccess || numDevices <= 0) {
        zend_throw_error(NULL, "No GPU device available or CUDA not enabled");
        return;
    }
    if (deviceId < 0 || deviceId >= (zend_long)numDevices) {
        zend_throw_error(NULL,
                         "Device " ZEND_LONG_FMT " does not exist "
                         "(valid range: 0.." ZEND_LONG_FMT ")",
                         deviceId, (zend_long)(numDevices - 1));
        return;
    }
    err = cudaSetDevice((int)deviceId);
    if (err != cudaSuccess) {
        zend_throw_error(NULL, "cudaSetDevice failed for device " ZEND_LONG_FMT ": %s",
                         deviceId, cudaGetErrorString(err));
        return;
    }
#else
    (void)deviceId;
    zend_throw_error(NULL, "No GPU device available or CUDA not enabled");
    return;
#endif
}

// @todo Indices conversion lose precision, we must convert it directly to a integer vector in C
//       without relying on ZVAL_TO_NDARRAY. We must apply the same for all other cases where a
//       PHP array of longs is converted to NDArray before being converted to a C integer.
ZEND_BEGIN_ARG_INFO(arginfo_reshape, 2)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO();
PHP_METHOD(NumPower, reshape) {
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
    ndarray_init_new_object(rtn, return_value);
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
                target = buffer_get(getObjectUuid(var));
                RETURN_STRING(NDArray_Print(target, 1));
            }
        }
        RETURN_STR(zend_print_zval_r_to_str(var, 0));
    } else {
        if (Z_TYPE_P(var) == IS_OBJECT) {
            zend_class_entry* classEntry = Z_OBJCE_P(var);
            if (!strcmp(classEntry->name->val, "NDArray")) {
                target = buffer_get(getObjectUuid(var));
                NDArray_Print(target, 0);
                RETURN_TRUE;
            }
        }
        zend_print_zval_r(var, 0);
        RETURN_TRUE;
    }
}

/**
 * @brief Validate the (dtype, device) parameter pair shared by every
 *        typed-and-deviced initializer.
 *
 * Resolves @p data_type to its canonical static pointer (NULL or empty
 * selects float32) and checks @p device against
 * `{NDARRAY_DEVICE_CPU, NDARRAY_DEVICE_GPU}`. On a non-CUDA build the
 * GPU device is rejected loudly here so callers never receive an
 * NDArray with uninitialised on-device storage. Throws a catchable
 * `\Error` and returns 0 on any failure.
 *
 * @param[in]  data_type  Optional dtype alias; NULL/empty → float32.
 * @param[in]  device     Long device id from `Z_PARAM_LONG`.
 * @param[out] out_dtype  Canonical static dtype pointer.
 * @param[out] out_device Validated device id (int).
 * @return 1 on success, 0 on validation failure (Error in flight).
 */
static int
ndarray_parse_dtype_device(const char *data_type, zend_long device,
                           const char **out_dtype, int *out_device) {
    const char *ndarray_dtype = NDARRAY_TYPE_FLOAT32;
    if (data_type != NULL && *data_type != '\0') {
        ndarray_dtype = type_canonicalize(data_type);
        if (ndarray_dtype == NULL) {
            zend_throw_error(NULL,
                "Invalid data type '%s'. Supported: float4, float8, float16, "
                "float32, float64, float128, int8, uint8, int16, uint16, "
                "int32, uint32, int64, uint64", data_type);
            return 0;
        }
    }

    if (device != NDARRAY_DEVICE_CPU && device != NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL,
            "Invalid device %lld. Use 0 (CPU) or 1 (GPU).",
            (long long) device);
        return 0;
    }
#ifndef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL,
            "GPU device requested but the extension was built without CUDA support.");
        return 0;
    }
#endif

    *out_dtype  = ndarray_dtype;
    *out_device = (int) device;
    return 1;
}

/**
 * @brief Parse the (shape, dtype, device) parameter triplet shared by the
 *        typed-and-deviced shape initializers (`zeros`, `ones`, `full`).
 *
 * Wraps `ndarray_parse_dtype_device` with the shape-extraction half:
 * the shape zval is routed through `ZVAL_TO_NDARRAY` and every entry is
 * checked against negativity. On success the caller owns @p out_shape
 * (freshly emalloc'd, transferred into the NDArray builder).
 *
 * @param[in]  shape_zval   PHP shape value (array / scalar / NDArray).
 * @param[in]  data_type    Optional dtype alias.
 * @param[in]  device       Long device id.
 * @param[out] out_dtype    Canonical static dtype pointer.
 * @param[out] out_device   Validated device id.
 * @param[out] out_shape    Newly-allocated `int[*out_ndim]`.
 * @param[out] out_ndim     Number of dimensions.
 * @return 1 on success, 0 on validation failure (Error in flight).
 */
static int
ndarray_parse_typed_shape(zval *shape_zval, const char *data_type,
                          zend_long device, const char **out_dtype,
                          int *out_device, int **out_shape, int *out_ndim) {
    if (!ndarray_parse_dtype_device(data_type, device, out_dtype, out_device)) {
        return 0;
    }

    NDArray *nda = ZVAL_TO_NDARRAY(shape_zval);
    if (nda == NULL) {
        return 0;
    }
    int ndim = NDArray_NUMELEMENTS(nda);
    /* Allocate at least one int slot even for ndim == 0 — `emalloc(0)` is
       valid but yields an opaque token, while a real slot keeps ASAN /
       Valgrind happy and gives the NDArray builder a non-NULL dimensions
       pointer to store. */
    int *shape = emalloc(sizeof(int) * (ndim > 0 ? (size_t) ndim : 1));
    for (int i = 0; i < ndim; i++) {
        double raw = (double) NDArray_F32DATA(nda)[i];
        if (raw < 0.0) {
            efree(shape);
            /* `CHECK_INPUT_AND_FREE` releases the temporary NDArray built
               by `ZVAL_TO_NDARRAY` only for scalar / array sources; an
               existing NDArray passed as shape keeps its refcount, so a
               bare `NDArray_FREE` here would drop the user's live array
               into a dangling buffer slot. */
            CHECK_INPUT_AND_FREE(shape_zval, nda);
            zend_throw_error(NULL, "negative dimensions are not allowed");
            return 0;
        }
        shape[i] = (int) raw;
    }
    CHECK_INPUT_AND_FREE(shape_zval, nda);

    *out_shape = shape;
    *out_ndim  = ndim;
    return 1;
}

/**
 * @brief `NumPower::zeros(shape, dtype = "float32", device = 0): NDArray`.
 *
 * Allocates a zero-initialised NDArray of the requested shape and dtype on
 * the requested device. For `device == 1` (GPU) the buffer is allocated
 * directly in VRAM via `cudaMalloc` and zeroed in place with `cudaMemset`,
 * so no host buffer is ever materialised; the host->device copy that
 * `->gpu()` would normally pay is skipped entirely.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_zeros, 0)
ZEND_ARG_INFO(0, shape)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, zeros) {
    zval *shape_zval;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(shape_zval)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    NDArray *rtn = NDArray_Zeros(shape, ndim, ndarrayDataType, parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Factory methods always return an NDArray, even for a 0-D shape (`[]`),
       matching numpy's `zeros(())` → `array(0.)` contract. Collapsing 0-D to
       a host primitive via `ndarray_init_new_object` would defeat the entire
       point of an explicit `device == GPU` call: the user paid for the VRAM
       allocation and expects an NDArray they can `->dump()`, feed back into
       further GPU ops, etc. Use the always-NDArray installer instead. */
    ndarray_install_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * @brief `NumPower::identity(size, dtype = "float32", device = 0): NDArray`.
 *
 * Builds a `size × size` identity matrix on the requested device with
 * the dtype-appropriate representation of 1 on the main diagonal and
 * zeros elsewhere. For `device == 1` (GPU) the backing buffer is
 * allocated directly in VRAM (`cudaMalloc` + `cudaMemset(0)`) and the
 * diagonal is written by a single `cudaMemcpy2D` H2D with a stride of
 * `(size + 1) * elsize` — the destination matrix itself never traverses
 * host memory; only the small (`≤ 16 * size`-byte) host source seed
 * carrying the dtype's "1" representation does.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_identity, 0)
ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, identity) {
    zend_long size;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_LONG(size)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    /* Range-check `size` against `int` *before* truncating: a value that
       overflows `int` would otherwise wrap to a small / negative number
       and silently produce the wrong matrix or trip the negative-dim
       guard with a misleading error. */
    if (size < 0 || size > INT_MAX) {
        zend_throw_error(NULL,
            "identity: size %lld is out of range (must be 0..%d)",
            (long long) size, INT_MAX);
        return;
    }

    const char *ndarrayDataType;
    int parsed_device;
    if (!ndarray_parse_dtype_device(dataType, device,
                                    &ndarrayDataType, &parsed_device)) {
        return;
    }

    NDArray *rtn = NDArray_Identity((int) size, ndarrayDataType, parsed_device);
    if (rtn == NULL) {
        return;
    }
    ndarray_install_object(rtn, return_value);
}

/* Forward declarations: the polymorphic zval→numeric coercion helpers
   live further down in this file (next to `arange`). `normal` was added
   later so it needs the declarations to call them. */
static int coerce_zval_to_double(zval *z, const char *op, const char *name,
                                 double *out_dst);
static int coerce_zval_to_uint64(zval *z, const char *op, const char *name,
                                 uint64_t *out_dst);
static int coerce_zval_to_fp128(zval *z, const char *op, const char *name,
                                ndarray_fp128_t *out_dst);

/**
 * @brief Initialise an `NDArrayNormalSpec` with the N(0, 1) defaults that
 *        match @p type's arithmetic kind.
 *
 * The kind comes from `NDArray_NormalKindFor`; the matching union arm is
 * set to loc=0 and scale=1 in that kind's native precision (`fp128` zero
 * and `fp128` one for FP128, native `uint64_t` 0 and 1 for UINT64,
 * `double` 0.0 and 1.0 for DOUBLE). Callers that want non-default
 * loc/scale (`normal`, `truncatedNormal`) follow up with
 * `ndarray_normal_spec_overlay` to pull values from user-supplied zvals;
 * the `standardNormal` entry point uses the defaults as-is.
 *
 * @param[out] spec Spec to initialise; `kind` and the active union arm are set.
 * @param[in]  type Canonical dtype string (selects the kind).
 */
static void
ndarray_normal_spec_defaults(NDArrayNormalSpec *spec, const char *type) {
    spec->kind = NDArray_NormalKindFor(type);
    switch (spec->kind) {
        case NDARRAY_NORMAL_KIND_FP128:
            spec->v.f128.loc   = NDARRAY_FP128_ZERO();
            spec->v.f128.scale = NDARRAY_FP128_FROM_I64(1);
            break;
        case NDARRAY_NORMAL_KIND_UINT64:
            spec->v.u64.loc   = 0;
            spec->v.u64.scale = 1;
            break;
        case NDARRAY_NORMAL_KIND_DOUBLE:
        default:
            spec->kind       = NDARRAY_NORMAL_KIND_DOUBLE;
            spec->v.d.loc    = 0.0;
            spec->v.d.scale  = 1.0;
            break;
    }
}

/**
 * @brief Overlay user-supplied @p loc_zv / @p scale_zv onto a spec whose
 *        defaults were initialised by `ndarray_normal_spec_defaults`,
 *        then validate that scale is non-negative.
 *
 * Each non-NULL zval is coerced into the active kind's native scalar
 * type via the matching `coerce_zval_to_*` helper. A failed coercion
 * leaves a catchable `Error` in flight and returns 0; the caller is
 * expected to release any shape buffer it allocated and return.
 *
 * After successful coercion, scale is checked for negativity — a
 * negative standard deviation is mathematically nonsensical (σ² ≥ 0
 * by definition) and would also feed cuRAND a value outside its
 * documented `stddev > 0` precondition on the GPU paths. `scale == 0`
 * is permitted (degenerate distribution where every sample equals
 * loc); the CPU fillers handle it trivially and cuRAND empirically
 * returns the loc value. The UINT64 arm needs no check (unsigned 64
 * is always ≥ 0 by type).
 *
 * @param[in,out] spec      Spec carrying the dtype-aware defaults; the
 *                          active union arm is overwritten on success.
 * @param[in]     op        Method name prefix used in error messages
 *                          (e.g. "normal", "truncatedNormal").
 * @param[in]     loc_zv    Optional user-supplied loc (NULL keeps the
 *                          default).
 * @param[in]     scale_zv  Optional user-supplied scale (NULL keeps the
 *                          default).
 * @return 1 on success, 0 on type rejection or negative scale
 *         (Error in flight).
 */
static int
ndarray_normal_spec_overlay(NDArrayNormalSpec *spec, const char *op,
                            zval *loc_zv, zval *scale_zv) {
    switch (spec->kind) {
        case NDARRAY_NORMAL_KIND_FP128:
            if (loc_zv != NULL &&
                !coerce_zval_to_fp128(loc_zv, op, "loc",
                                       &spec->v.f128.loc)) {
                return 0;
            }
            if (scale_zv != NULL &&
                !coerce_zval_to_fp128(scale_zv, op, "scale",
                                       &spec->v.f128.scale)) {
                return 0;
            }
            if (NDARRAY_FP128_LT(spec->v.f128.scale, NDARRAY_FP128_ZERO())) {
                zend_throw_error(NULL,
                    "%s: scale must be non-negative", op);
                return 0;
            }
            break;
        case NDARRAY_NORMAL_KIND_UINT64:
            if (loc_zv != NULL &&
                !coerce_zval_to_uint64(loc_zv, op, "loc",
                                        &spec->v.u64.loc)) {
                return 0;
            }
            if (scale_zv != NULL &&
                !coerce_zval_to_uint64(scale_zv, op, "scale",
                                        &spec->v.u64.scale)) {
                return 0;
            }
            /* uint64 scale is unsigned — no negativity check needed. */
            break;
        case NDARRAY_NORMAL_KIND_DOUBLE:
        default:
            if (loc_zv != NULL &&
                !coerce_zval_to_double(loc_zv, op, "loc",
                                        &spec->v.d.loc)) {
                return 0;
            }
            if (scale_zv != NULL &&
                !coerce_zval_to_double(scale_zv, op, "scale",
                                        &spec->v.d.scale)) {
                return 0;
            }
            /* `!(scale >= 0.0)` also rejects NaN. */
            if (!(spec->v.d.scale >= 0.0)) {
                zend_throw_error(NULL,
                    "%s: scale must be non-negative", op);
                return 0;
            }
            break;
    }
    return 1;
}

/**
 * @brief Initialise an `NDArrayUniformSpec` with U([0, 1)) defaults that
 *        match @p type's arithmetic kind.
 *
 * Parallel of `ndarray_normal_spec_defaults`: the kind comes from
 * `NDArray_UniformKindFor`; the matching union arm is set to `low = 0`
 * and `high = 1` in that kind's native precision (`fp128` zero / one
 * for FP128, native `uint64_t` 0 / 1 for UINT64, `double` 0.0 / 1.0 for
 * DOUBLE). The defaults match numpy's `random.uniform` defaults; callers
 * that want non-default bounds (the `NumPower::uniform` PHP entry point)
 * then call `ndarray_uniform_spec_overlay` to pull values from
 * user-supplied zvals.
 *
 * @param[out] spec Spec to initialise; `kind` and the active union arm are set.
 * @param[in]  type Canonical dtype string (selects the kind).
 */
static void
ndarray_uniform_spec_defaults(NDArrayUniformSpec *spec, const char *type) {
    spec->kind = NDArray_UniformKindFor(type);
    switch (spec->kind) {
        case NDARRAY_UNIFORM_KIND_FP128:
            spec->v.f128.low  = NDARRAY_FP128_ZERO();
            spec->v.f128.high = NDARRAY_FP128_FROM_I64(1);
            break;
        case NDARRAY_UNIFORM_KIND_UINT64:
            spec->v.u64.low  = 0;
            spec->v.u64.high = 1;
            break;
        case NDARRAY_UNIFORM_KIND_DOUBLE:
        default:
            spec->kind        = NDARRAY_UNIFORM_KIND_DOUBLE;
            spec->v.d.low     = 0.0;
            spec->v.d.high    = 1.0;
            break;
    }
}

/**
 * @brief Overlay user-supplied @p low_zv / @p high_zv onto a spec whose
 *        defaults were initialised by `ndarray_uniform_spec_defaults`.
 *
 * Parallel of `ndarray_normal_spec_overlay`: each non-NULL zval is
 * coerced into the active kind's native scalar type via the matching
 * `coerce_zval_to_*` helper. A failed coercion leaves a catchable
 * `Error` in flight and returns 0; the caller is expected to release
 * any shape buffer it allocated and return.
 *
 * @param[in,out] spec     Spec carrying the dtype-aware defaults; the
 *                         active union arm is overwritten on success.
 * @param[in]     op       Method name prefix used in error messages
 *                         (e.g. "uniform").
 * @param[in]     low_zv   Optional user-supplied low (NULL keeps the
 *                         default).
 * @param[in]     high_zv  Optional user-supplied high (NULL keeps the
 *                         default).
 * @return 1 on success, 0 on type rejection (Error in flight).
 */
static int
ndarray_uniform_spec_overlay(NDArrayUniformSpec *spec, const char *op,
                              zval *low_zv, zval *high_zv) {
    switch (spec->kind) {
        case NDARRAY_UNIFORM_KIND_FP128:
            if (low_zv != NULL &&
                !coerce_zval_to_fp128(low_zv, op, "low",
                                       &spec->v.f128.low)) {
                return 0;
            }
            if (high_zv != NULL &&
                !coerce_zval_to_fp128(high_zv, op, "high",
                                       &spec->v.f128.high)) {
                return 0;
            }
            break;
        case NDARRAY_UNIFORM_KIND_UINT64:
            if (low_zv != NULL &&
                !coerce_zval_to_uint64(low_zv, op, "low",
                                        &spec->v.u64.low)) {
                return 0;
            }
            if (high_zv != NULL &&
                !coerce_zval_to_uint64(high_zv, op, "high",
                                        &spec->v.u64.high)) {
                return 0;
            }
            break;
        case NDARRAY_UNIFORM_KIND_DOUBLE:
        default:
            if (low_zv != NULL &&
                !coerce_zval_to_double(low_zv, op, "low",
                                        &spec->v.d.low)) {
                return 0;
            }
            if (high_zv != NULL &&
                !coerce_zval_to_double(high_zv, op, "high",
                                        &spec->v.d.high)) {
                return 0;
            }
            break;
    }
    return 1;
}

/**
 * @brief `NumPower::normal(shape, loc = 0.0, scale = 1.0, dtype = "float32", device = 0): NDArray`.
 *
 * Generates an NDArray of the requested shape filled with samples drawn
 * from N(@p loc, @p scale^2). The result dtype and residency device are
 * caller-selectable; on `device == 1` (GPU) the buffer is allocated
 * directly in VRAM (via `NDArray_Empty`) and populated by cuRAND
 * (`curandGenerateNormal` / `curandGenerateNormalDouble`) plus
 * `cuda_cast_*` quantisation for non-fp32/fp64 dtypes — no full-size
 * host staging of the result. For `dtype == float128` on GPU the values
 * are computed in true double-double arithmetic on device via a custom
 * `cuda_normal_dd_affine` kernel so the user's fp128 loc/scale are
 * preserved bit-for-bit through the (hi, lo) DD layout.
 *
 * `loc` and `scale` accept `int | float | string`. The string form is
 * the only precision-loss-free route for the wide dtypes:
 * `float128` strings parse via `strtoflt128` (or the DD fallback),
 * `uint64` strings via `strtoull`. Every other dtype is coerced through
 * `double`, which represents each smaller dtype's range exactly.
 *
 * `device` is `0` for CPU (default) or `1` for GPU. The same `NUMPOWER_CPU`
 * / `NUMPOWER_CUDA` constants the rest of the API uses.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_normal, 0, 0, 1)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, loc)
    ZEND_ARG_INFO(0, scale)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, normal) {
    zval *shape_zval;
    zval *loc_zv   = NULL;
    zval *scale_zv = NULL;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_ZVAL(shape_zval)
    Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(loc_zv)
        Z_PARAM_ZVAL(scale_zv)
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    /* Defaults are dtype-aware: `loc = 0` / `scale = 1` are encoded in
       the same precision as any explicit arg so the three dispatch paths
       stay symmetric. `ndarray_normal_spec_overlay` then merges any
       user-supplied loc/scale onto those defaults. */
    NDArrayNormalSpec spec;
    ndarray_normal_spec_defaults(&spec, ndarrayDataType);
    if (!ndarray_normal_spec_overlay(&spec, "normal", loc_zv, scale_zv)) {
        efree(shape);
        return;
    }

    NDArray *rtn = NDArray_Normal(&spec, shape, ndim, ndarrayDataType,
                                  parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Match every other typed factory (zeros / ones / arange) — always
       hand back an NDArray, even for a 0-D shape, so the GPU-residency
       contract isn't broken by a primitive-zval collapse. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::truncatedNormal(shape, loc = 0.0, scale = 1.0, dtype = "float32", device = 0): NDArray`.
 *
 * Same contract as `NumPower::normal()` except every per-element draw
 * is rejection-bounded so the result lies in `[loc - 2σ, loc + 2σ]`.
 * The implementation reuses the `NDArrayNormalSpec` discriminator and
 * the `coerce_zval_to_*` helpers — `loc` and `scale` accept
 * `int | float | string`, with the string form being the only
 * precision-loss-free path for the wide dtypes (`float128` via
 * `strtoflt128` / DD fallback; `uint64` via `strtoull`).
 *
 * GPU paths (`device == 1`) build the result directly in VRAM via the
 * new `cuda_truncated_normal_f32` / `cuda_truncated_normal_f64`
 * rejection-sample kernels; for `float128` GPU the values go through a
 * standardised f64 truncated stream + `cuda_normal_dd_affine` so the
 * user's fp128 loc/scale survive bit-for-bit through the (hi, lo) DD
 * layout. See `NDArray_TruncatedNormal` for the per-dtype dispatch
 * table.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_truncated_normal, 0, 0, 1)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, loc)
    ZEND_ARG_INFO(0, scale)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, truncatedNormal) {
    zval *shape_zval;
    zval *loc_zv   = NULL;
    zval *scale_zv = NULL;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_ZVAL(shape_zval)
    Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(loc_zv)
        Z_PARAM_ZVAL(scale_zv)
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    NDArrayNormalSpec spec;
    ndarray_normal_spec_defaults(&spec, ndarrayDataType);
    if (!ndarray_normal_spec_overlay(&spec, "truncatedNormal",
                                      loc_zv, scale_zv)) {
        efree(shape);
        return;
    }

    NDArray *rtn = NDArray_TruncatedNormal(&spec, shape, ndim, ndarrayDataType,
                                            parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Hand back an NDArray even for a 0-D shape — matches every other
       typed factory and preserves the GPU-residency contract. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::randomBinomial(shape, n, p, dtype = "float32", device = 0): NDArray`.
 *
 * Generates an NDArray of the requested shape filled with samples
 * drawn from a Binomial distribution `B(n, p)` — each element is the
 * count of successes across @p n independent Bernoulli trials with
 * per-trial success probability @p p. The result dtype and residency
 * device are caller-selectable; on `device == 1` (GPU) the buffer is
 * allocated directly in VRAM (via `NDArray_Empty`) and populated by a
 * custom per-thread cuRAND kernel — no full-size host staging of the
 * result.
 *
 * @p n is parsed as `double` to preserve the legacy contract and then
 * cast to `int` internally (the algorithm itself uses an integer trial
 * count). The PHP entry point validates `n >= 0`, `0 <= p <= 1`, and
 * `n <= INT_MAX`, surfacing clear errors for any out-of-range input.
 *
 * `device` is `0` for CPU (default) or `1` for GPU. The same
 * `NUMPOWER_CPU` / `NUMPOWER_CUDA` constants the rest of the API uses.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_binomial, 0, 0, 3)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, n)
    ZEND_ARG_INFO(0, p)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, randomBinomial) {
    zval *shape_zval;
    double n = 0.0;
    double p = 0.0;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(3, 5)
        Z_PARAM_ZVAL(shape_zval)
        Z_PARAM_DOUBLE(n)
        Z_PARAM_DOUBLE(p)
    Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    /* Validate `n` and `p` before any allocation — these are integer
       distribution parameters and out-of-range values would produce
       silent garbage downstream. */
    if (!(n >= 0.0)) {
        /* `!(n >= 0.0)` also rejects NaN. */
        zend_throw_error(NULL,
            "randomBinomial: n must be a non-negative real number");
        return;
    }
    if (n > (double)INT_MAX) {
        zend_throw_error(NULL,
            "randomBinomial: n must be ≤ INT_MAX (%d)", INT_MAX);
        return;
    }
    if (!(p >= 0.0 && p <= 1.0)) {
        zend_throw_error(NULL,
            "randomBinomial: p must be in [0.0, 1.0]");
        return;
    }

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    NDArray *rtn = NDArray_Binomial(shape, ndim, (int)n, (float)p,
                                     ndarrayDataType, parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Hand back an NDArray even for a 0-D shape — matches every other
       typed factory and preserves the GPU-residency contract. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::standardNormal(shape, dtype = "float32", device = 0): NDArray`.
 *
 * Convenience entry point for the standard normal distribution N(0, 1):
 * a fixed `loc = 0`, `scale = 1` pair routed through the same
 * `NDArray_Normal` dispatcher that powers `NumPower::normal()`. Every
 * supported dtype is honoured — for `float128` and `uint64` the defaults
 * are encoded in their native precision via `ndarray_normal_spec_defaults`
 * so loc/scale survive bit-for-bit into the dtype-aware fill path.
 *
 * For `device == 1` (GPU) the destination buffer is allocated directly
 * in VRAM (`NDArray_Empty` → `vmalloc` → `cudaMalloc`) and populated by
 * cuRAND (`curandGenerateNormal` / `curandGenerateNormalDouble`) plus
 * `cuda_cast_*` quantisation for non-fp32/fp64 dtypes — there is no
 * full-size host staging of the result. For `dtype == 'float128'` on GPU
 * the values flow through a custom `cuda_normal_dd_affine` kernel that
 * performs the affine in true double-double arithmetic so the (hi, lo)
 * DD layout is preserved.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_standard_normal, 0, 0, 1)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, standardNormal) {
    zval *shape_zval;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(shape_zval)
    Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    NDArrayNormalSpec spec;
    ndarray_normal_spec_defaults(&spec, ndarrayDataType);

    NDArray *rtn = NDArray_Normal(&spec, shape, ndim, ndarrayDataType,
                                  parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Hand back an NDArray even for a 0-D shape — matches every other
       typed factory and preserves the GPU-residency contract. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::poisson(shape, lam = 1.0, dtype = "float32", device = 0): NDArray`.
 *
 * Generates an NDArray of the requested shape filled with samples drawn
 * from a Poisson distribution with rate parameter @p lam. The result
 * dtype and residency device are caller-selectable; on `device == 1`
 * (GPU) the buffer is allocated directly in VRAM (via `NDArray_Empty`)
 * and populated by cuRAND (`curandGeneratePoisson`) + `cuda_cast_*`
 * quantisation for non-uint32 dtypes — no full-size host staging of
 * the result.
 *
 * `lam` accepts `int | float | string`. The string form is the
 * precision-loss-free parser path consistent with the rest of the
 * random-family entry points; for Poisson the rate is internally a
 * `double` (the algorithm's numerical core is fp64 on both CPU and
 * GPU), so wide-range strings funnel through `strtod` and are limited
 * to fp64 precision.
 *
 * CPU sampling uses Knuth's multiplicative method for `lam < 30` and
 * Hörmann's PTRS rejection algorithm for `lam ≥ 30` — same threshold
 * as numpy's `random_poisson`, so the per-sample cost stays bounded
 * regardless of the rate. The legacy implementation used `expf(-lam)`
 * which underflowed to `0` for `lam ≥ 88` and caused the inner loop
 * to spin forever — that bug is fixed.
 *
 * `device` is `0` for CPU (default) or `1` for GPU. The same
 * `NUMPOWER_CPU` / `NUMPOWER_CUDA` constants the rest of the API uses.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_poisson, 0, 0, 1)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, lam)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, poisson) {
    zval *shape_zval;
    zval *lam_zv = NULL;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(shape_zval)
    Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(lam_zv)
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    /* Poisson's rate is always a `double` internally; the int|float|string
       coercion path matches every other random-family entry point. */
    double lam = 1.0;
    if (lam_zv != NULL && !coerce_zval_to_double(lam_zv, "poisson", "lam",
                                                  &lam)) {
        efree(shape);
        return;
    }
    if (!(lam >= 0.0)) {
        /* `!(lam >= 0.0)` catches negative values *and* NaN — the
           algorithm constants (`exp(-lam)`, `sqrt(lam)`) would otherwise
           produce silent NaN propagation. */
        efree(shape);
        zend_throw_error(NULL,
            "poisson: lam must be a non-negative real number");
        return;
    }

    NDArray *rtn = NDArray_Poisson(lam, shape, ndim, ndarrayDataType,
                                    parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Hand back an NDArray even for a 0-D shape — matches every other
       typed factory and preserves the GPU-residency contract. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::uniform(shape, low = 0.0, high = 1.0, dtype = "float32", device = 0): NDArray`.
 *
 * Generates an NDArray of the requested shape filled with samples drawn
 * from the continuous uniform distribution `U([low, high))`. The result
 * dtype and residency device are caller-selectable; on `device == 1`
 * (GPU) the buffer is allocated directly in VRAM (via `NDArray_Empty`)
 * and populated by cuRAND (`curandGenerateUniform` /
 * `curandGenerateUniformDouble`) plus `cuda_cast_*` quantisation for
 * non-fp32/fp64 dtypes — no full-size host staging of the result. For
 * `dtype == float128` on GPU the values are computed in true
 * double-double arithmetic on device via a custom
 * `cuda_uniform_dd_affine` kernel so the user's fp128 low/high are
 * preserved bit-for-bit through the (hi, lo) DD layout.
 *
 * `low` and `high` accept `int | float | string`. The string form is
 * the only precision-loss-free route for the wide dtypes:
 * `float128` strings parse via `strtoflt128` (or the DD fallback),
 * `uint64` strings via `strtoull`. Every other dtype is coerced through
 * `double`, which represents each smaller dtype's range exactly.
 *
 * The result range is `[low, high)` — closed at @c low, open at
 * @c high. cuRAND's native uniform output is `(0, 1]`, so the GPU
 * affine reflects via `1 - u` before scaling so the closed endpoint
 * lands at @c low (matches numpy's contract).
 *
 * `device` is `0` for CPU (default) or `1` for GPU. The same
 * `NUMPOWER_CPU` / `NUMPOWER_CUDA` constants the rest of the API uses.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_uniform, 0, 0, 1)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, low)
    ZEND_ARG_INFO(0, high)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, uniform) {
    zval *shape_zval;
    zval *low_zv  = NULL;
    zval *high_zv = NULL;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_ZVAL(shape_zval)
    Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(low_zv)
        Z_PARAM_ZVAL(high_zv)
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    /* Defaults are dtype-aware: `low = 0` / `high = 1` are encoded in
       the same precision as any explicit arg so the three dispatch
       paths stay symmetric. `ndarray_uniform_spec_overlay` then merges
       any user-supplied low/high onto those defaults. */
    NDArrayUniformSpec spec;
    ndarray_uniform_spec_defaults(&spec, ndarrayDataType);
    if (!ndarray_uniform_spec_overlay(&spec, "uniform", low_zv, high_zv)) {
        efree(shape);
        return;
    }

    NDArray *rtn = NDArray_Uniform(&spec, shape, ndim, ndarrayDataType,
                                    parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* Hand back an NDArray even for a 0-D shape — matches every other
       typed factory and preserves the GPU-residency contract. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::diag(target, dtype = "float32", device = 0): NDArray`.
 *
 * Dual-mode like numpy's `diag`:
 *  - **1-D input** → builds an `N×N` diagonal matrix.
 *  - **2-D input** → extracts the main diagonal as a 1-D vector of
 *    `min(rows, cols)` elements.
 *
 * The result lives in the requested @c dtype on the requested @c device.
 * When the input doesn't already match, it is cast and / or moved to a
 * fresh copy that is freed before this function returns; the diagonal
 * traffic itself is a single `cudaMemcpy2D` D2D call on GPU (see
 * `NDArray_Diag`).
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_diag, 0, 0, 1)
    ZEND_ARG_INFO(0, target)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, diag) {
    zval *target;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(target)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    if (!ndarray_parse_dtype_device(dataType, device,
                                    &ndarrayDataType, &parsed_device)) {
        return;
    }

    NDArray *nda = ZVAL_TO_NDARRAY(target);
    if (nda == NULL) {
        return;
    }

    NDArray *rtn = NDArray_Diag(nda, ndarrayDataType, parsed_device);

    /* Release the temporary NDArray created by ZVAL_TO_NDARRAY only when
       the source zval is a scalar/array (an existing NDArray object
       reference keeps its refcount). */
    CHECK_INPUT_AND_FREE(target, nda);
    if (rtn == NULL) {
        return;
    }
    ndarray_install_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * @brief `NumPower::full(shape, fill_value, dtype = "float32", device = 0): NDArray`.
 *
 * Allocates an NDArray of the requested shape / dtype / device and fills
 * every element with @c fill_value encoded into the target dtype. The
 * fill value may be passed as `int`, `float`, `bool`, or `string`. The
 * string form is the only way to express the full range of `float128`,
 * `int64`, and `uint64` (values outside PHP's native long / double range
 * stay byte-correct).
 *
 * For `device == 1` (GPU) the buffer is allocated directly in VRAM via
 * `cudaMalloc` and populated by `cuda_fill_bytes` — a doubling
 * device-to-device broadcast loop. Only the one-element seed value
 * (≤ 16 bytes) crosses the PCIe bus, so host RAM stays `O(elsize)`
 * regardless of the tensor size.
 *
 * Pre-existing bugs fixed in this refactor:
 *   - the old implementation hardcoded float32 storage and broadcast a
 *     float-cast fill, corrupting any non-float32 dtype if one had ever
 *     been requested;
 *   - empty shape `[]` was rejected with a "non-empty array" error even
 *     though numpy returns `array(fv)` for `np.full((), fv)`;
 *   - the shape walk forced IS_LONG entries, then re-converted them
 *     through `NDArray_ToIntVector` (float-mantissa round-trip) which
 *     loses precision for dimensions ≥ 2²⁴.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_full, 0, 0, 2)
ZEND_ARG_INFO(0, shape)
ZEND_ARG_INFO(0, fill_value)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, full) {
    zval *shape_zval;
    zval *fill_value;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ZVAL(shape_zval)
        Z_PARAM_ZVAL(fill_value)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    /* Encode the fill value into the dtype's host representation once.
       16 bytes covers the widest dtype (fp128). On encoding failure the
       helper has already thrown — release the shape allocation we own
       to keep the request-scope buffer balanced. */
    char encoded[16];
    memset(encoded, 0, sizeof(encoded));
    if (!NDArray_EncodeZvalToDtype(fill_value, ndarrayDataType, encoded)) {
        efree(shape);
        return;
    }

    NDArray *rtn = NDArray_Full(shape, ndim, ndarrayDataType,
                                parsed_device, encoded);
    if (rtn == NULL) {
        return;
    }
    /* Match zeros() / ones(): factory methods always return an NDArray,
       even for a 0-D shape — preserves the GPU-residency contract. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief `NumPower::ones(shape, dtype = "float32", device = 0): NDArray`.
 *
 * Allocates an NDArray of the requested shape and dtype on the requested
 * device, with every element set to the dtype-appropriate representation
 * of 1. For `device == 1` (GPU) the buffer is allocated directly in VRAM
 * and populated via a doubling device-to-device broadcast (see
 * `cuda_fill_bytes`) — no full host-side staging buffer is allocated, only
 * the one-element seed value crosses the bus.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_ones, 0)
ZEND_ARG_INFO(0, shape)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, ones) {
    zval *shape_zval;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(shape_zval)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    int *shape;
    int ndim;
    if (!ndarray_parse_typed_shape(shape_zval, dataType, device,
                                   &ndarrayDataType, &parsed_device,
                                   &shape, &ndim)) {
        return;
    }

    NDArray *rtn = NDArray_Ones(shape, ndim, ndarrayDataType, parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* See `zeros()` above for the rationale — same contract: factory methods
       return an NDArray even for a 0-D shape, preserving the GPU residency
       contract when the caller passed `device == GPU`. */
    ndarray_install_object(rtn, return_value);
}

/**
 * @brief Coerce a PHP scalar @p z into a `double`.
 *
 * Accepts `int`, `float`, and numeric `string` inputs. The string path
 * uses `strtod`, matching PHP's standard numeric-string coercion. The
 * double @p out_dst is set only on success; on failure a catchable
 * `\Error` is in flight, prefixed with @p op so the caller's surface is
 * obvious in the error message (e.g. `"arange: stop must be …"`).
 *
 * Shared by every PHP method that accepts a polymorphic numeric scalar
 * (currently `arange`, `normal`).
 *
 * @param[in]  z       PHP scalar.
 * @param[in]  op      Method name prefix for the error message.
 * @param[in]  name    Argument label for the error message.
 * @param[out] out_dst Receives the coerced double on success.
 * @return 1 on success, 0 on a type rejection.
 */
static int
coerce_zval_to_double(zval *z, const char *op, const char *name, double *out_dst) {
    if (Z_TYPE_P(z) == IS_LONG) {
        *out_dst = (double) Z_LVAL_P(z);
        return 1;
    }
    if (Z_TYPE_P(z) == IS_DOUBLE) {
        *out_dst = Z_DVAL_P(z);
        return 1;
    }
    if (Z_TYPE_P(z) == IS_STRING) {
        *out_dst = strtod(Z_STRVAL_P(z), NULL);
        return 1;
    }
    zend_throw_error(NULL,
        "%s: %s must be int, float, or numeric string", op, name);
    return 0;
}

/**
 * @brief Coerce a PHP scalar @p z into an `int64_t`.
 *
 * IS_LONG is taken verbatim (PHP's native long is 64-bit signed on every
 * supported platform). IS_DOUBLE is cast through `(int64_t)`. IS_STRING
 * routes through `strtoll` so values outside PHP's long range stay
 * byte-correct.
 *
 * @param[in]  z       PHP scalar.
 * @param[in]  op      Method name prefix for the error message.
 * @param[in]  name    Argument label for the error message.
 * @param[out] out_dst Receives the int64 value on success.
 * @return 1 on success, 0 on type rejection (Error in flight).
 */
static int
coerce_zval_to_int64(zval *z, const char *op, const char *name, int64_t *out_dst) {
    if (Z_TYPE_P(z) == IS_LONG) {
        *out_dst = (int64_t) Z_LVAL_P(z);
        return 1;
    }
    if (Z_TYPE_P(z) == IS_DOUBLE) {
        *out_dst = (int64_t) Z_DVAL_P(z);
        return 1;
    }
    if (Z_TYPE_P(z) == IS_STRING) {
        *out_dst = (int64_t) strtoll(Z_STRVAL_P(z), NULL, 10);
        return 1;
    }
    zend_throw_error(NULL,
        "%s: %s must be int, float, or numeric string", op, name);
    return 0;
}

/**
 * @brief Coerce a PHP scalar @p z into a `uint64_t`.
 *
 * IS_LONG is range-checked (a negative value is rejected for unsigned).
 * IS_STRING routes through `strtoull` so `"18446744073709551615"` and
 * other values above `LLONG_MAX` survive intact.
 *
 * @param[in]  z       PHP scalar.
 * @param[in]  op      Method name prefix for the error message.
 * @param[in]  name    Argument label for the error message.
 * @param[out] out_dst Receives the uint64 value on success.
 * @return 1 on success, 0 on type rejection or negative-long.
 */
static int
coerce_zval_to_uint64(zval *z, const char *op, const char *name, uint64_t *out_dst) {
    if (Z_TYPE_P(z) == IS_LONG) {
        zend_long lv = Z_LVAL_P(z);
        if (lv < 0) {
            zend_throw_error(NULL,
                "%s: %s must be non-negative for uint64 dtype", op, name);
            return 0;
        }
        *out_dst = (uint64_t) lv;
        return 1;
    }
    if (Z_TYPE_P(z) == IS_DOUBLE) {
        double dv = Z_DVAL_P(z);
        if (dv < 0.0) {
            zend_throw_error(NULL,
                "%s: %s must be non-negative for uint64 dtype", op, name);
            return 0;
        }
        *out_dst = (uint64_t) dv;
        return 1;
    }
    if (Z_TYPE_P(z) == IS_STRING) {
        *out_dst = (uint64_t) strtoull(Z_STRVAL_P(z), NULL, 10);
        return 1;
    }
    zend_throw_error(NULL,
        "%s: %s must be int, float, or numeric string", op, name);
    return 0;
}

/**
 * @brief Coerce a PHP scalar @p z into `ndarray_fp128_t`.
 *
 * IS_STRING is the precision-loss-free path (via `strtoflt128` on glibc,
 * or the DD fallback). IS_LONG / IS_DOUBLE go through the dtype's
 * standard conversion helpers.
 *
 * @param[in]  z       PHP scalar.
 * @param[in]  op      Method name prefix for the error message.
 * @param[in]  name    Argument label for the error message.
 * @param[out] out_dst Receives the fp128 value on success.
 * @return 1 on success, 0 on type rejection.
 */
static int
coerce_zval_to_fp128(zval *z, const char *op, const char *name,
                     ndarray_fp128_t *out_dst) {
    if (Z_TYPE_P(z) == IS_STRING) {
        *out_dst = ndarray_string_to_fp128(Z_STRVAL_P(z));
        return 1;
    }
    if (Z_TYPE_P(z) == IS_LONG) {
        *out_dst = NDARRAY_FP128_FROM_I64((int64_t) Z_LVAL_P(z));
        return 1;
    }
    if (Z_TYPE_P(z) == IS_DOUBLE) {
        *out_dst = ndarray_double_to_fp128(Z_DVAL_P(z));
        return 1;
    }
    zend_throw_error(NULL,
        "%s: %s must be int, float, or numeric string", op, name);
    return 0;
}

/**
 * @brief `NumPower::arange(stop, start = 0, step = 1, dtype = "float32", device = 0): NDArray`.
 *
 * Generates a 1-D NDArray whose values follow `a[i] = start + i * step`
 * for `i` in `[0, ceil((stop - start) / step))`. The sign of `step` must
 * be consistent with the (start, stop) interval; otherwise an empty
 * array is returned (numpy behaviour).
 *
 * The first three parameters accept `int`, `float`, and numeric `string`
 * inputs. The string form is the only loss-free route for the wide
 * dtypes: `float128` strings flow through `strtoflt128` / the DD parser,
 * `uint64` strings through `strtoull`, and `int64` strings through
 * `strtoll`. For every other dtype the value is coerced through `double`
 * (which represents every smaller dtype's range exactly).
 *
 * For `device == 1` (GPU) the destination matrix is allocated directly
 * in VRAM via `NDArray_Empty`; the closed-form values are computed in a
 * transient host scratch and then shipped to the device via
 * `NDArray_TypedH2D` (which handles fp128's host→DD layout conversion).
 * Writing the elements via a custom on-device kernel would be a pure-
 * VRAM alternative; the current shape of the code keeps host involvement
 * to the `n * elsize` scratch which is freed immediately afterwards.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arange, 0, 0, 1)
    ZEND_ARG_INFO(0, stop)
    ZEND_ARG_INFO(0, start)
    ZEND_ARG_INFO(0, step)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arange) {
    zval *stop_zv;
    zval *start_zv = NULL;
    zval *step_zv = NULL;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_ZVAL(stop_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(start_zv)
        Z_PARAM_ZVAL(step_zv)
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();

    const char *ndarrayDataType;
    int parsed_device;
    if (!ndarray_parse_dtype_device(dataType, device,
                                    &ndarrayDataType, &parsed_device)) {
        return;
    }

    /* Defaults are dtype-aware: `start = 0` and `step = 1` are encoded
       in the same precision as the explicit args so the four dispatch
       paths stay symmetric. */
    NDArrayArangeSpec spec;
    spec.kind = NDArray_ArangeKindFor(ndarrayDataType);

    switch (spec.kind) {
        case NDARRAY_ARANGE_KIND_FP128: {
            spec.v.f128.start = NDARRAY_FP128_ZERO();
            spec.v.f128.step  = NDARRAY_FP128_FROM_I64(1);
            if (!coerce_zval_to_fp128(stop_zv, "arange", "stop", &spec.v.f128.stop)) return;
            if (start_zv != NULL &&
                !coerce_zval_to_fp128(start_zv, "arange", "start", &spec.v.f128.start)) return;
            if (step_zv != NULL &&
                !coerce_zval_to_fp128(step_zv,  "arange", "step",  &spec.v.f128.step)) return;
            break;
        }
        case NDARRAY_ARANGE_KIND_INT64: {
            spec.v.i64.start = 0;
            spec.v.i64.step  = 1;
            if (!coerce_zval_to_int64(stop_zv, "arange", "stop", &spec.v.i64.stop)) return;
            if (start_zv != NULL &&
                !coerce_zval_to_int64(start_zv, "arange", "start", &spec.v.i64.start)) return;
            if (step_zv != NULL &&
                !coerce_zval_to_int64(step_zv,  "arange", "step",  &spec.v.i64.step)) return;
            break;
        }
        case NDARRAY_ARANGE_KIND_UINT64: {
            spec.v.u64.start = 0;
            spec.v.u64.step  = 1;
            if (!coerce_zval_to_uint64(stop_zv, "arange", "stop", &spec.v.u64.stop)) return;
            if (start_zv != NULL &&
                !coerce_zval_to_uint64(start_zv, "arange", "start", &spec.v.u64.start)) return;
            if (step_zv != NULL &&
                !coerce_zval_to_uint64(step_zv,  "arange", "step",  &spec.v.u64.step)) return;
            break;
        }
        case NDARRAY_ARANGE_KIND_DOUBLE:
        default: {
            spec.kind = NDARRAY_ARANGE_KIND_DOUBLE;
            spec.v.d.start = 0.0;
            spec.v.d.step  = 1.0;
            if (!coerce_zval_to_double(stop_zv, "arange", "stop", &spec.v.d.stop)) return;
            if (start_zv != NULL &&
                !coerce_zval_to_double(start_zv, "arange", "start", &spec.v.d.start)) return;
            if (step_zv != NULL &&
                !coerce_zval_to_double(step_zv,  "arange", "step",  &spec.v.d.step)) return;
            break;
        }
    }

    NDArray *rtn = NDArray_Arange(&spec, ndarrayDataType, parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* arange always returns at least a 1-D array (possibly empty), so the
       0-D scalar collapse in `ndarray_init_new_object` never triggers;
       use `ndarray_install_object` for uniformity with the other typed
       factories. */
    ndarray_install_object(rtn, return_value);
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
        ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(output, return_value);
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
    ndarray_init_new_object(output, return_value);
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
    ndarray_init_new_object(output, return_value);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * @brief Infer the loss-free dtype for a bare numeric string scalar.
 *
 * Bare strings carry no peer-NDArray context, so the dispatcher decides
 * the dtype from the string's content. The rule is conservative and
 * picks the widest dtype that can hold the value without rounding —
 * strings are precisely the intake path callers use to escape PHP's
 * native long/double range:
 *  - any decimal point, exponent marker (`e`/`E`), or `inf`/`infinity`/
 *    `nan` token (case-insensitive) → `float128` (the only dtype that
 *    holds wide decimal literals end-to-end);
 *  - non-negative integer literal whose magnitude exceeds UINT64_MAX
 *    → `float128` (escalate to keep the precision rather than saturate
 *    at `2^64 - 1`);
 *  - non-negative integer literal with magnitude > INT64_MAX (i.e.
 *    > 19 digits, or 19 digits exceeding the INT64_MAX prefix) →
 *    `uint64` (the only dtype that holds the full 64-bit range);
 *  - negative integer literal → `int64` (negatives can't fit `uint64`,
 *    and `INT64_MIN`'s magnitude is bounded by 19 digits);
 *  - otherwise → `int64` (every magnitude that fits both signed and
 *    unsigned, including zero).
 *
 * Strict syntactic validation runs in the same pass: the function
 * rejects any literal that is not a syntactically complete numeric
 * value (e.g. `"0xff"`, `"abc"`, `"1.5.5"`, `"1.5a"`, `"1,5"`,
 * `"1.5e"`, `"  -  1.5"`) by returning NULL. The caller throws
 * `"Numeric string expected"`. This prevents the silent-zero coercion
 * that `strtoll`/`strtoull`/`strtoflt128` would otherwise return when
 * given a partial or malformed literal.
 *
 * Magnitude is compared digit-by-digit against the INT64_MAX /
 * UINT64_MAX strings, which sidesteps the strtoull / errno boundary
 * and works correctly even with leading-zero or leading-`+` literals.
 * Whitespace at either end is tolerated (mirrors `strtod`).
 *
 * @param[in] str  String content (need not be NUL-terminated within
 *                 @p len bytes).
 * @param[in] len  Length in bytes.
 * @return Canonical dtype string ("float128" / "int64" / "uint64") or
 *         NULL when @p str is empty, all-whitespace, or syntactically
 *         malformed (caller throws).
 */
static const char *ndarray_infer_dtype_from_string(const char *str, size_t len)
{
    if (str == NULL || len == 0) {
        return NULL;
    }
    /* Trim leading whitespace (strtod-style). */
    size_t i = 0;
    while (i < len && (str[i] == ' ' || str[i] == '\t' ||
                       str[i] == '\n' || str[i] == '\r')) {
        i++;
    }
    /* Trim trailing whitespace. */
    size_t end = len;
    while (end > i && (str[end - 1] == ' ' || str[end - 1] == '\t' ||
                       str[end - 1] == '\n' || str[end - 1] == '\r')) {
        end--;
    }
    if (i == end) {
        return NULL;
    }

    /* Detect inf / infinity / nan tokens (case-insensitive). The token
       must consume the entire (trimmed, possibly signed) literal — the
       silent-tail-junk problem applies here too. */
    size_t sign_off = i;
    if (str[sign_off] == '+' || str[sign_off] == '-') {
        sign_off++;
        if (sign_off == end) return NULL;  /* "+" / "-" alone */
    }
    size_t tail_len = end - sign_off;
    if (tail_len == 3 || tail_len == 8) {
        char low[9] = {0};
        for (size_t k = 0; k < tail_len; k++) {
            low[k] = (char)(str[sign_off + k] | 0x20);
        }
        if ((tail_len == 3 && (memcmp(low, "inf", 3) == 0 ||
                               memcmp(low, "nan", 3) == 0)) ||
            (tail_len == 8 && memcmp(low, "infinity", 8) == 0)) {
            return "float128";
        }
    }

    /* Strict numeric-literal scan over str[i..end). Tracks whether the
       literal contains a fractional / exponent part (→ fp128) and where
       the magnitude digits live (for the int64-vs-uint64 split). Any
       character that does not fit the grammar fails the scan. */
    size_t k          = i;
    int    has_sign   = 0;
    int    is_neg     = 0;
    if (k < end && (str[k] == '+' || str[k] == '-')) {
        has_sign = 1;
        is_neg   = (str[k] == '-');
        k++;
    }
    /* Mantissa integer part. */
    size_t mant_int_start = k;
    while (k < end && str[k] >= '0' && str[k] <= '9') k++;
    size_t mant_int_len = k - mant_int_start;

    int saw_dot = 0;
    size_t frac_len = 0;
    if (k < end && str[k] == '.') {
        saw_dot = 1;
        k++;
        size_t frac_start = k;
        while (k < end && str[k] >= '0' && str[k] <= '9') k++;
        frac_len = k - frac_start;
    }
    /* At least one digit in mantissa (integer or fractional). */
    if (mant_int_len == 0 && frac_len == 0) {
        (void)has_sign;
        return NULL;
    }

    int saw_exp = 0;
    if (k < end && (str[k] == 'e' || str[k] == 'E')) {
        saw_exp = 1;
        k++;
        if (k < end && (str[k] == '+' || str[k] == '-')) k++;
        size_t exp_start = k;
        while (k < end && str[k] >= '0' && str[k] <= '9') k++;
        if (k == exp_start) {
            /* Exponent marker without digits, e.g. "1.5e". */
            return NULL;
        }
    }
    /* Everything from i..end must have been consumed. */
    if (k != end) {
        return NULL;
    }

    /* Floating-point literal → fp128. */
    if (saw_dot || saw_exp) {
        return "float128";
    }

    /* Pure integer literal. */
    if (is_neg) {
        return "int64";
    }
    /* Magnitude check on the integer digits (skip leading zeros). */
    const char *p = str + mant_int_start;
    size_t      m = mant_int_len;
    while (m > 1 && *p == '0') { p++; m--; }

    static const char int64_max_str[]  = "9223372036854775807";   /* 19 digits */
    static const char uint64_max_str[] = "18446744073709551615";  /* 20 digits */
    if (m > 20) {
        /* Past UINT64_MAX — escalate to fp128 to keep precision rather
           than saturate the integer dtypes. */
        return "float128";
    }
    if (m == 20) {
        if (memcmp(p, uint64_max_str, 20) > 0) return "float128";
        return "uint64";
    }
    if (m == 19 && memcmp(p, int64_max_str, 19) > 0) {
        return "uint64";
    }
    return "int64";
}

/**
 * @brief Resolve the input zval of a unary op (`abs`, `exp`, `clip`, …)
 *        into a usable NDArray, honouring the bare-string-scalar intake.
 *
 * Three intake forms are accepted:
 *  - **NDArray / nested array / int / float**: routed through
 *    `ZVAL_TO_NDARRAY`. The returned NDArray is *borrowed* from the
 *    caller's zval — caller must release via `CHECK_INPUT_AND_FREE`
 *    (which is a no-op when the zval already wraps an NDArray and an
 *    `NDArray_FREE` otherwise).
 *  - **Numeric string**: dtype inferred from the literal via
 *    `ndarray_infer_dtype_from_string`, then encoded into a fresh 0-D
 *    NDArray on CPU via `ndarray_make_typed_scalar`. This is the only
 *    intake that lets a one-call expression carry full `float128` /
 *    `uint64` precision without first allocating an NDArray. The
 *    returned NDArray is *owned* by the caller — release via
 *    `NDArray_FREE`.
 *  - **Malformed / empty / whitespace-only string**: throws a clear PHP
 *    error citing the offending literal and returns NULL.
 *
 * Shared by every unary method that funnels through
 * `ndarray_run_simple_unary` and by `clip`, which has extra lo/hi
 * bound parameters but otherwise the same intake contract.
 *
 * @param[in]  array PHP zval supplied as the array argument.
 * @param[out] owned Receives 1 when the caller must release the
 *                   returned NDArray with `NDArray_FREE`, 0 when the
 *                   pair (zval, NDArray) must be released through
 *                   `CHECK_INPUT_AND_FREE`. Untouched on failure.
 * @return NDArray on success; NULL on validation failure (with a PHP
 *         exception in flight).
 */
static NDArray *ndarray_resolve_unary_input(zval *array, int *owned)
{
    if (Z_TYPE_P(array) == IS_STRING) {
        const char *dt = ndarray_infer_dtype_from_string(
            Z_STRVAL_P(array), Z_STRLEN_P(array));
        if (dt == NULL) {
            /* Differentiate the three failure modes so callers can spot
               typos quickly: empty literal, whitespace-only, or
               syntactically malformed (non-empty, non-whitespace). */
            const char *p = Z_STRVAL_P(array);
            size_t      n = Z_STRLEN_P(array);
            size_t      ws = 0;
            while (ws < n && (p[ws] == ' ' || p[ws] == '\t' ||
                              p[ws] == '\n' || p[ws] == '\r')) {
                ws++;
            }
            if (n == 0) {
                zend_throw_error(NULL,
                    "Numeric string expected, got an empty value.");
            } else if (ws == n) {
                zend_throw_error(NULL,
                    "Numeric string expected, got a whitespace-only value.");
            } else {
                zend_throw_error(NULL,
                    "Numeric string expected, got malformed literal: \"%s\".",
                    p);
            }
            return NULL;
        }
        NDArray *nda = ndarray_make_typed_scalar(array, dt);
        if (nda == NULL) {
            return NULL;
        }
        *owned = 1;
        return nda;
    }
    *owned = 0;
    return ZVAL_TO_NDARRAY(array);
}

/**
 * @brief Release the NDArray resolved by `ndarray_resolve_unary_input`,
 *        choosing the right free path for the ownership flag.
 *
 * Owned NDArrays come from `ndarray_make_typed_scalar` and are released
 * directly via `NDArray_FREE`. Borrowed NDArrays go through
 * `CHECK_INPUT_AND_FREE`, which is a no-op when the zval already wraps
 * an NDArray (so the user's reference survives) and an `NDArray_FREE`
 * when the zval was a literal array / int / float (so the transient
 * `ZVAL_TO_NDARRAY` buffer is returned to the pool).
 *
 * @param[in] array Original input zval.
 * @param[in] nda   NDArray previously returned by
 *                  `ndarray_resolve_unary_input`.
 * @param[in] owned The matching ownership flag.
 */
static void ndarray_release_unary_input(zval *array, NDArray *nda, int owned)
{
    if (owned) {
        NDArray_FREE(nda);
    } else {
        CHECK_INPUT_AND_FREE(array, nda);
    }
}

/**
 * @brief Run a typed unary op on a single zval argument and install
 *        the result.
 *
 * Covers every op enumerated by `NDArrayUnaryOp` that takes exactly
 * one NDArray input and no extra parameters: the basic family (`abs`,
 * `negative`, `positive`, `reciprocal`, `sign`, `sqrt`, `rsqrt`,
 * `square`, `sinc`), the transcendental family (`exp`, `exp2`,
 * `expm1`, `log`, `log1p`, `log2`, `log10`, `logb`), the
 * trigonometric / hyperbolic family (`sin`, `cos`, `tan`, `arcsin`,
 * `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`,
 * `arctanh`), the angle-conversion ops (`degrees`, `radians`), and
 * the rounding ops (`rint`, `fix`, `trunc`, `floor`, `ceil`).
 *
 * The clip op uses its own entry because of the lo / hi parameters;
 * `arctan2` (binary) and `round` (precision param) likewise still
 * ride bespoke entry points until the dispatcher grows
 * binary-unary / extra-arg support.
 *
 * Centralises the PHP-binding plumbing every unary method needs:
 *  - resolves the input zval to an NDArray via
 *    `ndarray_resolve_unary_input` (handles NDArray / array / scalar /
 *    numeric string, infers dtype for the string intake). String-scalar
 *    inputs always materialize a 0-D NDArray on CPU because there is no
 *    peer NDArray to anchor a device choice on — pre-existing GPU
 *    NDArrays passed as inputs still execute on GPU per the device
 *    contract below;
 *  - dispatches through `NDArray_TypedUnaryOp`, which keeps GPU inputs on
 *    GPU and matches CPU bit-for-bit on every supported dtype;
 *  - releases the resolved NDArray via `ndarray_release_unary_input`,
 *    routing to the right free path for owned vs borrowed inputs so no
 *    buffer slot survives the call;
 *  - installs the result via `ndarray_init_new_object`, which collapses
 *    a 0-D result to the dtype-correct PHP scalar.
 *
 * @param[in]  INTERNAL_FUNCTION_PARAMETERS Standard PHP entry point macros.
 * @param[in]  op                           Unary op selector.
 */
static void
ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAMETERS, NDArrayUnaryOp op) {
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();

    int nda_owned;
    NDArray *nda = ndarray_resolve_unary_input(array, &nda_owned);
    if (nda == NULL) {
        return;
    }

    NDArray *rtn = NDArray_TypedUnaryOp(op, nda, NULL, NULL);
    ndarray_release_unary_input(array, nda, nda_owned);
    if (rtn == NULL) {
        return;
    }
    ndarray_init_new_object(rtn, return_value);
}

/**
 * NumPower::abs
 *
 * Element-wise absolute value, preserving the input dtype (unsigned-int
 * inputs return a copy unchanged). For `INT_MIN` on signed ints the
 * result wraps modulo 2^N, matching NumPy.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_abs, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, abs) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ABS);
}

/**
 * NumPower::sin
 *
 * Element-wise `sin(x)`. Integer inputs widen to float32 (narrow) or
 * float64 (32/64-bit) per PyTorch widening; every floating-point dtype
 * is preserved. Computation stays on the input's device.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sin, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sin) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_SIN);
}

/**
 * NumPower::cos
 *
 * Element-wise `cos(x)`. Same dtype / device contract as `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_cos, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, cos) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_COS);
}

/**
 * NumPower::tan
 *
 * Element-wise `tan(x)`. Same dtype / device contract as `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_tan, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, tan) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_TAN);
}

/**
 * NumPower::arcsin
 *
 * Element-wise `arcsin(x)`. Same dtype / device contract as
 * `NumPower::sin`. Inputs outside [-1, 1] return NaN per IEEE 754.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arcsin, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arcsin) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ARCSIN);
}

/**
 * NumPower::rsqrt
 *
 * Element-wise reciprocal square root, `1 / sqrt(x)`. Integer inputs
 * widen to float (narrow ints → float32, 32/64-bit ints → float64);
 * float dtypes keep their dtype. Reaches every supported dtype on
 * both CPU and GPU.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_rsqrt, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, rsqrt) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_RSQRT);
}

/**
 * NumPower::arccos
 *
 * Element-wise `arccos(x)` (inverse cosine). Same dtype / device
 * contract as `NumPower::sin`. Inputs outside [-1, 1] return NaN
 * per IEEE 754; result range is [0, π].
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arccos, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arccos) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ARCCOS);
}

/**
 * NumPower::arctan
 *
 * Element-wise `arctan(x)`. Same dtype / device contract as
 * `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctan, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arctan) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ARCTAN);
}

/**
 * NumPower::arctan2
 *
 * Two-argument arctangent `atan2(x, y)` element-wise. Returns the
 * angle in radians between the positive x-axis and the point `(y, x)`,
 * choosing the quadrant from the signs of both args.
 *
 * Out of scope of the typed-unary refactor — `arctan2` is a binary op
 * and still rides the legacy `NDArray_Map1ND` / `cuda_float_arctan2`
 * path which assumes float32. See [[sin-cos-trig-dtype-bug]] follow-up.
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
    if (ndx == NULL) {
        return;
    }
    NDArray *ndy = ZVAL_TO_NDARRAY(y);
    if (ndy == NULL) {
        CHECK_INPUT_AND_FREE(x, ndx);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * NumPower::degrees
 *
 * Element-wise radians → degrees conversion (multiplies by 180/π).
 * Integer inputs widen to float32 (narrow) / float64 (32/64-bit) per
 * PyTorch widening; every floating-point dtype is preserved.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_degrees, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, degrees) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_DEGREES);
}

/**
 * NumPower::sinh
 *
 * Element-wise `sinh(x)`. Same dtype / device contract as
 * `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sinh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sinh) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_SINH);
}

/**
 * NumPower::cosh
 *
 * Element-wise `cosh(x)`. Same dtype / device contract as
 * `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_cosh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, cosh) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_COSH);
}

/**
 * NumPower::tanh
 *
 * Element-wise `tanh(x)`. Same dtype / device contract as
 * `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_tanh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, tanh) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_TANH);
}

/**
 * NumPower::arcsinh
 *
 * Element-wise inverse hyperbolic sine. Same dtype / device contract
 * as `NumPower::sin`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arcsinh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arcsinh) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ARCSINH);
}

/**
 * NumPower::arccosh
 *
 * Element-wise inverse hyperbolic cosine. Same dtype / device
 * contract as `NumPower::sin`. Inputs `< 1` return NaN per IEEE 754.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arccosh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arccosh) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ARCCOSH);
}

/**
 * @brief `NumPower::fromImage($image, $channelLast = true, $dtype = "uint8",
 *        $device = 0): NDArray` — load a GD image into a typed NDArray.
 *
 * Pixel values are integers in [0, 255], so `uint8` is the default dtype —
 * 4× smaller than the legacy float32 storage. Callers who need a different
 * numeric range (e.g. normalized float pipelines feeding into ML) can pass
 * any of the 14 supported dtypes; non-uint8 dtypes get the raw 0..255
 * values encoded into their representation, ready for downstream arithmetic.
 * The legacy implementation produced HWC `[W, H, 3]` (width and height
 * swapped) — this version emits `[H, W, 3]` to match numpy/PIL/PyTorch.
 *
 * @param[in] execute_data PHP call frame.
 * @param[in] return_value zval to populate with the new `NDArray` object.
 *
 * @throws \Error When called against a build without GD; when the dtype is
 *                unknown; when the device is GPU and the build lacks CUDA.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_fromimage, 0, 0, 1)
    ZEND_ARG_INFO(0, image)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, channelLast, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "\"uint8\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, device, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, fromImage) {
    zval *image;
    bool channelLast = true;
    char *dataType = NULL;
    size_t dataTypeLen = 0;
    zend_long device = NDARRAY_DEVICE_CPU;

    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(image)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(channelLast)
        Z_PARAM_STRING(dataType, dataTypeLen)
        Z_PARAM_LONG(device)
    ZEND_PARSE_PARAMETERS_END();
#ifdef HAVE_GD
    if (Z_TYPE_P(image) != IS_OBJECT ||
        strcmp(ZSTR_VAL(Z_OBJ_P(image)->ce->name), "GdImage") != 0) {
        zend_throw_error(NULL,
            "NumPower::fromImage() expects a \\GdImage instance as the first argument.");
        return;
    }

    /* Default to uint8 — pixel values are 0..255 by definition. The
       ndarray_parse_dtype_device helper canonicalises the string and
       validates that GPU was requested only when CUDA is compiled in. */
    const char *ndarrayDataType;
    int parsed_device;
    const char *requested_dtype = (dataType != NULL && dataTypeLen > 0)
                                  ? dataType : "uint8";
    if (!ndarray_parse_dtype_device(requested_dtype, device,
                                    &ndarrayDataType, &parsed_device)) {
        return;
    }

    NDArray *rtn = NDArray_FromGD(image, channelLast, ndarrayDataType,
                                  parsed_device);
    if (rtn == NULL) {
        return;
    }
    /* fromImage always returns a 3-D NDArray, never a host scalar, so the
       always-NDArray installer is the right choice — it keeps the result
       chainable with `->gpu()` / `->reshape()` / etc. regardless of the
       (H, W) values. */
    ndarray_install_object(rtn, return_value);
#else
    (void)image; (void)channelLast; (void)dataType; (void)dataTypeLen;
    (void)device;
    zend_throw_error(NULL, "NumPower::fromImage() requires the extension to be built with GD support.");
#endif
}

/**
 * NumPower::arctanh
 *
 * Element-wise inverse hyperbolic tangent. Same dtype / device
 * contract as `NumPower::sin`. Inputs outside (-1, 1) return ±Inf
 * (at ±1) or NaN (outside) per IEEE 754.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctanh, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, arctanh) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_ARCTANH);
}

/**
 * NumPower::rint
 *
 * Element-wise IEEE 754 round-to-nearest-even (banker's rounding).
 * Dtype is preserved; integer inputs pass through unchanged.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_rint, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, rint) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_RINT);
}

/**
 * NumPower::fix
 *
 * Element-wise truncation toward zero (numpy's `np.fix`, equal to
 * `trunc` for IEEE 754). Dtype is preserved; integer inputs pass
 * through unchanged.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_fix, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, fix) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_FIX);
}

/**
 * NumPower::trunc
 *
 * Element-wise truncation toward zero. Dtype is preserved; integer
 * inputs pass through unchanged.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_trunc, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, trunc) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_TRUNC);
}

/**
 * NumPower::sinc
 *
 * Normalised sinc, `sin(π·x) / (π·x)` with `sinc(0) = 1`. Integer
 * inputs widen to float (narrow ints → float32, wide ints → float64);
 * float dtypes keep their dtype.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sinc, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sinc) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_SINC);
}

/**
 * NumPower::negative
 *
 * Element-wise unary minus. Preserves dtype; integer types wrap modulo
 * 2^N (matching NumPy and the binary `-$x` operator on NDArray).
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_negative, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, negative) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_NEGATIVE);
}

/**
 * NumPower::positive
 *
 * Unary plus: identity. Returns a copy with the same values and same
 * dtype. Distinct from `abs` (which folds negative inputs to their
 * magnitude) — the legacy implementation collapsed positive into abs;
 * the typed dispatcher restores NumPy's `np.positive` semantics.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_positive, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, positive) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_POSITIVE);
}

/**
 * NumPower::reciprocal
 *
 * `1 / x`. Integer inputs widen to float (narrow → float32, wide →
 * float64) per PyTorch's `result_type` rule for transcendental ops;
 * float dtypes keep their dtype.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_reciprocal, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, reciprocal) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_RECIPROCAL);
}


/**
 * NumPower::sign
 *
 * `-1` for negative, `0` for zero, `+1` for positive — cast back to
 * the source dtype. Unsigned-int inputs return 0 for the all-zero
 * element and `1` for every non-zero element.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sign, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sign) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_SIGN);
}

/**
 * NumPower::clip
 *
 * Element-wise clamp to `[min, max]`, preserving the input dtype.
 * `min` / `max` accept `int`, `float`, or `string`; the string path
 * is the only loss-free intake for `float128` (~34 digit literals)
 * and `uint64` (values > `PHP_INT_MAX`). Strings are parsed in the
 * input array's dtype so the GPU `dd` kernel sees the full DD pair
 * and the CPU `__float128` kernel sees the full 113-bit binary
 * fraction.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_clip, 0, 0, 3)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, min)
    ZEND_ARG_INFO(0, max)
ZEND_END_ARG_INFO()

/**
 * @brief Stringify a numeric zval (int / float / string) into a freshly
 *        emalloc'd buffer the caller must `efree`.
 *
 * Strings pass through unchanged so `float128` / `uint64` precision is
 * retained byte-for-byte. Integers stringify via `ZEND_LONG_FMT`; floats
 * via `%.17g` (round-trip precision for fp64; precision beyond fp64 is
 * unavoidably lost when the user passes a PHP float — they should pass
 * a string for `float128` precision). Throws and returns NULL for any
 * other zval type (bool / null / array / object).
 *
 * @param[in] z     Candidate bound zval.
 * @param[in] which Diagnostic label ("min" / "max") for the thrown error.
 * @return Caller-owned `emalloc`'d C string, or NULL on type rejection
 *         (PHP exception in flight).
 */
static char *
ndarray_clip_bound_to_string(zval *z, const char *which) {
    if (Z_TYPE_P(z) == IS_STRING) {
        return estrdup(Z_STRVAL_P(z));
    }
    char buf[64];
    if (Z_TYPE_P(z) == IS_LONG) {
        snprintf(buf, sizeof(buf), ZEND_LONG_FMT, Z_LVAL_P(z));
        return estrdup(buf);
    }
    if (Z_TYPE_P(z) == IS_DOUBLE) {
        snprintf(buf, sizeof(buf), "%.17g", Z_DVAL_P(z));
        return estrdup(buf);
    }
    zend_throw_error(NULL,
        "NumPower::clip: '%s' must be int, float, or string.", which);
    return NULL;
}

PHP_METHOD(NumPower, clip) {
    zval *array, *min_zv, *max_zv;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(array)
        Z_PARAM_ZVAL(min_zv)
        Z_PARAM_ZVAL(max_zv)
    ZEND_PARSE_PARAMETERS_END();

    char *min_str = ndarray_clip_bound_to_string(min_zv, "min");
    if (min_str == NULL) return;
    char *max_str = ndarray_clip_bound_to_string(max_zv, "max");
    if (max_str == NULL) { efree(min_str); return; }

    /* Bare numeric string `$array` is accepted here too: the shared
       helper infers fp128 / uint64 / int64 from the literal and builds
       a 0-D scalar via `ndarray_make_typed_scalar`. */
    int nda_owned;
    NDArray *nda = ndarray_resolve_unary_input(array, &nda_owned);
    if (nda == NULL) { efree(min_str); efree(max_str); return; }

    NDArray *rtn = NDArray_TypedUnaryOp(NDARRAY_UNOP_CLIP, nda,
                                         min_str, max_str);
    efree(min_str);
    efree(max_str);
    ndarray_release_unary_input(array, nda, nda_owned);
    if (rtn == NULL) return;
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
            double v = NDArray_Reduce_Mean(nda);
            CHECK_INPUT_AND_FREE(array, nda);
            RETURN_DOUBLE(v);
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
            double v = NDArray_Reduce_Mean(nda);
            CHECK_INPUT_AND_FREE(array, nda);
            RETURN_DOUBLE(v);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * NumPower::ceil
 *
 * Element-wise smallest integer not less than `x`. Dtype is
 * preserved; integer inputs pass through unchanged.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_ceil, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, ceil) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_CEIL);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * NumPower::floor
 *
 * Element-wise largest integer not greater than `x`. Dtype is
 * preserved; integer inputs pass through unchanged.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_floor, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, floor) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_FLOOR);
}

/**
 * NumPower::radians
 *
 * Element-wise degrees → radians conversion (multiplies by π/180).
 * Integer inputs widen to float32 (narrow) / float64 (32/64-bit) per
 * PyTorch widening; every floating-point dtype is preserved.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_radians, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, radians) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_RADIANS);
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
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_SQRT);
}

/**
 * NumPower::square
 *
 * Element-wise `x * x`, preserving the input dtype. Integer dtypes
 * wrap modulo 2^N (matching PyTorch and the binary `$x ** 2` path).
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_square, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, square) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_SQUARE);
}

/**
 * NumPower::exp
 *
 * Element-wise `e^x`. Integer inputs widen to float32 (narrow ints) or
 * float64 (32/64-bit ints) per PyTorch widening; every other dtype is
 * preserved. Computation stays on the input's device — GPU inputs are
 * served by typed `cuda_exp_*` kernels (fp32 / fp64 / fp16 / dd for
 * fp128) without CPU staging.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_exp, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, exp) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_EXP);
}

/**
 * NumPower::exp2
 *
 * Element-wise `2^x`. Integer inputs widen to float32 (narrow ints)
 * or float64 (32/64-bit ints) per PyTorch widening; every other dtype
 * is preserved.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_exp2, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, exp2) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_EXP2);
}

/**
 * NumPower::expm1
 *
 * Element-wise `e^x − 1` at higher precision than `exp(x) − 1` near
 * zero. Same dtype/device contract as `NumPower::exp`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_expm1, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, expm1) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_EXPM1);
}

/**
 * NumPower::log
 *
 * Element-wise natural logarithm `ln(x)`. Same dtype/device contract
 * as `NumPower::exp`; non-positive inputs return -inf or NaN per
 * IEEE 754.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_LOG);
}

/**
 * NumPower::logb
 *
 * Element-wise floating-point binary exponent `logb(x)` = floor(log2(|x|))
 * for normal numbers. Same dtype/device contract as `NumPower::exp`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_logb, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, logb) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_LOGB);
}

/**
 * NumPower::log10
 *
 * Element-wise base-10 logarithm `log₁₀(x)`. Same dtype/device contract
 * as `NumPower::exp`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log10, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log10) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_LOG10);
}

/**
 * NumPower::log1p
 *
 * Element-wise `ln(1 + x)` at higher precision than `log(1 + x)` near
 * zero. Same dtype/device contract as `NumPower::exp`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log1p, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log1p) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_LOG1P);
}

/**
 * NumPower::log2
 *
 * Element-wise base-2 logarithm `log₂(x)`. Same dtype/device contract
 * as `NumPower::exp`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log2, 0, 0, 1)
ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, log2) {
    ndarray_run_simple_unary(INTERNAL_FUNCTION_PARAM_PASSTHRU,
                             NDARRAY_UNOP_LOG2);
}

/**
 * @brief Shared PHP entry-point body for the six binary arithmetic methods.
 *
 * Every method (`add`, `subtract`, `multiply`, `divide`, `pow`, `mod`)
 * differs only in the opcode passed to `ndarray_arith_dispatch`. The
 * dispatcher handles weak-scalar promotion for IS_LONG/IS_DOUBLE/IS_STRING/
 * IS_TRUE/IS_FALSE (including loss-free `float128` / `int64` / `uint64`
 * string parsing), broadcasting, typed kernel dispatch on the correct
 * device, and dtype-correct return-value installation.
 *
 * @param[in]  opcode       Arithmetic opcode (ZEND_ADD / SUB / …).
 * @param[in]  execute_data PHP call frame (provides the two operands).
 * @param[out] return_value zval receiving the result.
 */
static void php_ndarray_arith_method(zend_uchar opcode, INTERNAL_FUNCTION_PARAMETERS)
{
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    (void)ndarray_arith_dispatch(opcode, a, b, return_value);
}

/**
 * @brief `NumPower::subtract(a, b): NDArray|int|float|string` — element-wise
 *        subtraction with broadcasting and weak-scalar (IS_STRING-capable)
 *        promotion. See `php_ndarray_arith_method`.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_subtract, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, subtract) {
    php_ndarray_arith_method(ZEND_SUB, INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

/**
 * @brief `NumPower::mod(a, b): NDArray|int|float|string` — element-wise
 *        remainder. Shares the dispatcher with the other binary ops.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_mod, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, mod) {
    php_ndarray_arith_method(ZEND_MOD, INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

/**
 * @brief `NumPower::pow(a, b): NDArray|int|float|string` — element-wise
 *        power. Shares the dispatcher with the other binary ops.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_pow, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, pow) {
    php_ndarray_arith_method(ZEND_POW, INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

/**
 * @brief `NumPower::multiply(a, b): NDArray|int|float|string` — element-wise
 *        multiplication. Shares the dispatcher with the other binary ops.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_multiply, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, multiply) {
    php_ndarray_arith_method(ZEND_MUL, INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

/**
 * @brief `NumPower::divide(a, b): NDArray|int|float|string` — element-wise
 *        true division (matches PyTorch: integer operands divide to float).
 *        Shares the dispatcher with the other binary ops.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_divide, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, divide) {
    php_ndarray_arith_method(ZEND_DIV, INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

/**
 * @brief `NumPower::add(a, b): NDArray|int|float|string` — element-wise
 *        addition. Shares the dispatcher with the other binary ops.
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_add, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, add) {
    php_ndarray_arith_method(ZEND_ADD, INTERNAL_FUNCTION_PARAM_PASSTHRU);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    zval *a;
    zval *source, *destination;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(source)
        Z_PARAM_ZVAL(destination)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        zend_throw_error(NULL, "Invalid NDArray provided.");
        return;
    }

    int ndim = NDArray_NDIM(nda);
    int src_size, dest_size;

    int *src = zval_parameter_to_normalized_axis_argument(source, "source", ndim, &src_size);
    if (src == NULL) {
        return;
    }

    int *dest = zval_parameter_to_normalized_axis_argument(destination, "destination", ndim, &dest_size);
    if (dest == NULL) {
        efree(src);
        return;
    }

    if (src_size != dest_size) {
        zend_throw_error(NULL, "`source` and `destination` must have the same number of elements.");
        efree(src);
        efree(dest);
        return;
    }

    NDArray *result = ndarray_moveaxis(nda, src, dest, src_size);
    efree(src);
    efree(dest);

    if (result == NULL) {
        return;
    }

    ndarray_init_new_object(result, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_ndarray_devicesync, 0)
ZEND_END_ARG_INFO()
/**
 * @brief NumPower::syncDevice() — block until queued GPU work finishes.
 *
 * Block until every CUDA kernel and asynchronous copy queued on the
 * current device has finished. This is the PHP-level entry point that
 * userland code can call before reading GPU results in a tight loop or
 * before taking a wall-clock benchmark; it mirrors PyTorch's
 * `torch.cuda.synchronize()`.
 *
 * When the extension is built without CUDA the call is a no-op so that
 * portable scripts can invoke it unconditionally. CUDA runtime errors
 * (sticky errors from earlier asynchronous launches included) surface
 * here as a PHP `\Error`.
 *
 * @throws \Error If `cudaDeviceSynchronize` reports a runtime error.
 */
PHP_METHOD(NumPower, syncDevice) {
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
#ifdef HAVE_CUBLAS
    /* If the toolkit is linked but no GPU is visible (CI container, dev
       machine without driver, etc.) treat the call as a no-op so that
       portable scripts can sprinkle syncDevice() unconditionally,
       matching the documented contract for the CPU-only build. */
    int numDevices = 0;
    cudaError_t probe = cudaGetDeviceCount(&numDevices);
    if (probe != cudaSuccess || numDevices <= 0) {
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        zend_throw_error(NULL, "cudaDeviceSynchronize failed: %s",
                         cudaGetErrorString(err));
        return;
    }
#endif
}

ZEND_BEGIN_ARG_INFO(arginfo_ndarray_rc, 0)
ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, rc) {
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    RETVAL_LONG(Z_REFCOUNT_P(a));
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

    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
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
    ndarray_init_new_object(rtn, return_value);
}

/**
 * NumPower::sum
 */
/**
 * @brief `NumPower::sum(a, axis = null): NDArray|int|float|string` — sum of
 *        elements, optionally over a single axis.
 *
 * `$axis === null` reduces every element to a 0-D NDArray collapsed by
 * `ndarray_init_new_object` into a dtype-correct PHP scalar — `string` for
 * `float128` / `uint64` (preserves the wide-dtype precision; `RETURN_DOUBLE`
 * would silently round past 2⁵³), `int` for the remaining integer dtypes,
 * `float` for the remaining floats.
 *
 * Integer `$axis` reduces along that axis; the result preserves @p a's
 * dtype and device. Negative indices count from the end (`-1` is the last
 * axis), matching numpy. The reduction stays on GPU for GPU inputs via the
 * typed binop kernels in `NDArray_Reduce_Axis`.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sum, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, sum) {
    zval *a;
    zend_long axis = 0;
    zend_bool axis_is_null = 1;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG_OR_NULL(axis, axis_is_null)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) return;

    NDArray *rtn = axis_is_null
        ? NDArray_Reduce_Sum_AsNDArray(nda)
        : NDArray_Reduce_Axis(nda, (int)axis, ND_AXIS_RED_SUM);
    CHECK_INPUT_AND_FREE(a, nda);
    if (rtn == NULL) return;
    rtn->uuid = -1;
    ndarray_init_new_object(rtn, return_value);
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
        value = NDArray_Reduce_Min(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    ndarray_init_new_object(rtn, return_value);
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
        value = NDArray_Reduce_Max(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    ndarray_init_new_object(rtn, return_value);
}

/**
 * @brief `NumPower::prod(a, axis = null): NDArray|int|float|string` — product
 *        of elements, optionally over a single axis.
 *
 * Mirrors `sum()` in every respect — dtype preservation, GPU residency,
 * dtype-correct scalar return, negative-axis support — but accumulates via
 * multiplication. See the `sum()` docblock for the per-dtype return-type
 * contract.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_prod, 0, 0, 1)
ZEND_ARG_INFO(0, a)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, prod) {
    zval *a;
    zend_long axis = 0;
    zend_bool axis_is_null = 1;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG_OR_NULL(axis, axis_is_null)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) return;

    NDArray *rtn = axis_is_null
        ? NDArray_Reduce_Prod_AsNDArray(nda)
        : NDArray_Reduce_Axis(nda, (int)axis, ND_AXIS_RED_PROD);
    CHECK_INPUT_AND_FREE(a, nda);
    if (rtn == NULL) return;
    rtn->uuid = -1;
    ndarray_init_new_object(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_ndarray_array, 0)
ZEND_ARG_INFO(0, a)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 0, "float64")
ZEND_END_ARG_INFO()
PHP_METHOD(NumPower, array) {
    NDArray *nda = NULL;
    zval *a;
    char *dataType = NULL;
    size_t dataTypeLen;
    const char *ndarrayDataType;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
    Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dataType, dataTypeLen)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() < 2) {
        dataType = "float32";
        dataTypeLen = sizeof("float32") - 1;
    }

    if      (!strcmp(dataType, "float4"))   ndarrayDataType = NDARRAY_TYPE_FLOAT4;
    else if (!strcmp(dataType, "float8"))   ndarrayDataType = NDARRAY_TYPE_FLOAT8;
    else if (!strcmp(dataType, "float16"))  ndarrayDataType = NDARRAY_TYPE_FLOAT16;
    else if (!strcmp(dataType, "float32"))  ndarrayDataType = NDARRAY_TYPE_FLOAT32;
    else if (!strcmp(dataType, "float64"))  ndarrayDataType = NDARRAY_TYPE_FLOAT64;
    else if (!strcmp(dataType, "float128")) ndarrayDataType = NDARRAY_TYPE_FLOAT128;
    else if (!strcmp(dataType, "int8"))     ndarrayDataType = NDARRAY_TYPE_INT8;
    else if (!strcmp(dataType, "uint8"))    ndarrayDataType = NDARRAY_TYPE_UINT8;
    else if (!strcmp(dataType, "int16"))    ndarrayDataType = NDARRAY_TYPE_INT16;
    else if (!strcmp(dataType, "uint16"))   ndarrayDataType = NDARRAY_TYPE_UINT16;
    else if (!strcmp(dataType, "int32"))    ndarrayDataType = NDARRAY_TYPE_INT32;
    else if (!strcmp(dataType, "uint32"))   ndarrayDataType = NDARRAY_TYPE_UINT32;
    else if (!strcmp(dataType, "int64"))    ndarrayDataType = NDARRAY_TYPE_INT64;
    else if (!strcmp(dataType, "uint64"))   ndarrayDataType = NDARRAY_TYPE_UINT64;
    else {
        zend_throw_error(NULL,
            "Invalid data type '%s'. Supported: float4, float8, float16, float32, float64, "
            "float128, int8, uint8, int16, uint16, int32, uint32, int64, uint64", dataType);
        return;
    }

    if (Z_TYPE_P(a) == IS_OBJECT) {
        zend_class_entry *ce = Z_OBJCE_P(a);
        if (instanceof_function(ce, phpsci_ce_NDArray)) {
            ZVAL_COPY(return_value, a);
            return;
        }
    }

    // NDArrayFactory_createFromZval adds the array to the buffer (uuid is set)
    nda = NDArrayFactory_createFromZval(a, ndarrayDataType);
    if (nda == NULL) {
        return;
    }
    object_init_ex(return_value, phpsci_ce_NDArray);
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(return_value), 0), NDArray_UUID(nda));
}

/**
 * Convert a variadic list of slice-spec zvals into NDArray pointers, invoke
 * NDArray_Slice, and release any temporaries auto-created from PHP scalars
 * or arrays. Shared by the instance and static slice methods so they use
 * identical plumbing.
 *
 * On any conversion error the function frees everything already allocated
 * and returns NULL with a PHP exception in flight; the caller must only
 * forward the NULL.
 *
 * @param src           Source NDArray; not retained by this call.
 * @param idx_args      Variadic argument array as produced by Z_PARAM_VARIADIC.
 * @param num_idx_args  Number of slice-spec zvals in `idx_args`.
 * @return              Freshly-allocated slice NDArray (uuid == -1) on
 *                      success, NULL if any conversion or slice failed.
 */
static NDArray *
php_ndarray_slice_run(NDArray *src, zval *idx_args, int num_idx_args) {
    NDArray **indices = emalloc(sizeof(NDArray *) * num_idx_args);

    for (int j = 0; j < num_idx_args; j++) {
        indices[j] = ZVAL_TO_NDARRAY(&idx_args[j]);
        if (indices[j] == NULL) {
            for (int k = 0; k < j; k++) {
                CHECK_INPUT_AND_FREE(&idx_args[k], indices[k]);
            }
            efree(indices);
            return NULL;
        }
    }

    NDArray *rtn = NDArray_Slice(src, indices, num_idx_args);

    /* CHECK_INPUT_AND_FREE only releases temporaries that were auto-created
       from PHP scalars/arrays; existing NDArray inputs keep their refcount,
       which is exactly what we want so the user's index variable survives. */
    for (int j = 0; j < num_idx_args; j++) {
        CHECK_INPUT_AND_FREE(&idx_args[j], indices[j]);
    }
    efree(indices);
    return rtn;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_slice, 0, 0, IS_MIXED, 0)
    ZEND_ARG_VARIADIC_TYPE_INFO(0, arg, IS_MIXED, 0)
ZEND_END_ARG_INFO()

/**
 * NDArray::slice(...$indices) — MUTATES $this in place.
 *
 * The underlying NDArray pointed to by $this is replaced with the slice
 * result. The PHP object identity is preserved (the same uuid → buffer slot
 * keeps holding $this), so any other PHP reference to the array sees the
 * mutation. Chained calls work because the method returns $this for non-0-D
 * results.
 *
 * For a 0-D result the method returns a dtype-correct scalar
 * (int / float / string per NDArray_ScalarToZval's rules), and $this is
 * still mutated to a 0-D NDArray.
 *
 * For a side-effect-free version, use the static NumPower::slice().
 *
 * @return NDArray|int|float|string $this on N-D, scalar on 0-D.
 */
PHP_METHOD(NDArray, slice) {
    zend_object *obj      = Z_OBJ_P(ZEND_THIS);
    zval        *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray     *ndarray  = ZVALUUID_TO_NDARRAY(obj_uuid);

    zval *arg;
    int   num_inputed_args = 0;
    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_VARIADIC('+', arg, num_inputed_args)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *rtn = php_ndarray_slice_run(ndarray, arg, num_inputed_args);
    if (rtn == NULL) {
        return;
    }

    /* Install `rtn` into $this's buffer slot atomically, then release the
       previous occupant. buffer_replace updates rtn->uuid for us and takes
       the global lock on ZTS builds so concurrent buffer_get/free calls
       never observe a torn slot. */
    int slot = (int) Z_LVAL_P(obj_uuid);
    NDArray *prev = buffer_replace(slot, rtn);
    NDArray_FREE(prev);

    if (NDArray_NDIM(rtn) > 0) {
        /* Chainable: return $this so `$a->slice(...)->slice(...)` works. */
        ZVAL_OBJ_COPY(return_value, obj);
    } else {
        /* 0-D: dtype-aware scalar (int for integer dtypes, string for
           float128/uint64, float otherwise). $this is still a 0-D NDArray
           after the call, mirroring numpy's "you sliced down to a scalar"
           state. */
        NDArray_ScalarToZval(rtn, return_value);
    }
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_numpower_slice, 0, 0, IS_MIXED, 0)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_VARIADIC_TYPE_INFO(0, indices, IS_MIXED, 0)
ZEND_END_ARG_INFO()

/**
 * NumPower::slice($array, ...$indices) — returns a NEW NDArray.
 *
 * Pure-function counterpart of NDArray::slice(): the source `$array` is
 * never mutated. Accepts the same wide input set as every other NumPower
 * static — NDArray, nested PHP array, or scalar — and the same
 * `int | [] | [start,stop] | [start,stop,step]` slice-spec grammar.
 *
 * Both the instance and static forms dispatch to the same underlying
 * NDArray_Slice algorithm, so per-device performance and edge-case behaviour
 * are identical.
 *
 * @return NDArray|int|float|string Fresh NDArray on N-D, dtype-correct
 *                                  scalar on 0-D (string for float128 and
 *                                  uint64, int for integer dtypes, float
 *                                  otherwise).
 */
PHP_METHOD(NumPower, slice) {
    zval *array_zv;
    zval *idx_args;
    int   num_idx_args = 0;

    ZEND_PARSE_PARAMETERS_START(2, -1)
        Z_PARAM_ZVAL(array_zv)
        Z_PARAM_VARIADIC('+', idx_args, num_idx_args)
    ZEND_PARSE_PARAMETERS_END();

    NDArray *src = ZVAL_TO_NDARRAY(array_zv);
    if (src == NULL) {
        return;
    }

    NDArray *rtn = php_ndarray_slice_run(src, idx_args, num_idx_args);
    CHECK_INPUT_AND_FREE(array_zv, src);

    if (rtn == NULL) {
        return;
    }
    /* ndarray_init_new_object handles both wrappings:
       - ndim  > 0 → wraps `rtn` in a fresh NDArray PHP object.
       - ndim == 0 → routes through NDArray_ScalarToZval (dtype-aware) and
                      frees `rtn`. */
    ndarray_init_new_object(rtn, return_value);
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
 * @brief NDArray::count() — implements the Countable interface.
 *
 * Returns the length of axis 0 (the leading dimension), which is the number
 * of sub-arrays produced by iterating with foreach. This is the value that
 * PHP's count() built-in returns when called on an NDArray, since the class
 * implements Countable: count($a) === $a->count() by construction.
 *
 * Semantics by rank:
 *   - 0-D scalar (shape []): returns 0. There is no axis 0 to enumerate,
 *     so the count matches an empty PHP array — consistent with shape()
 *     returning [] for the same array.
 *   - N-D (N >= 1): returns shape[0]. Cost is O(1) regardless of dtype,
 *     total element count, or device — count() reads only the cached
 *     dimension metadata, so it never touches the buffer or triggers a
 *     CPU/GPU transfer, and no memory is allocated.
 *
 * @return long Number of elements along axis 0, or 0 for 0-D scalars.
 */
PHP_METHOD(NDArray, count) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    if (NDArray_NDIM(ndarray) == 0) {
        RETURN_LONG(0);
    }
    RETURN_LONG(NDArray_SHAPE(ndarray)[0]);
}

/**
 * @brief NDArray::current() — Iterator value at the current axis-0 index.
 *
 * Returns a dtype-correct PHP scalar for 1-D source (int for int8..int64 and
 * uint8..uint32, string for uint64 / float128, float for the rest) and an
 * NDArray sub-view for N-D source. For 0-D source — or when the iterator has
 * advanced past the end — returns NULL, matching PHP's standard Iterator
 * convention (PHP's foreach machinery checks valid() first, but direct calls
 * are kept safe).
 *
 * The sub-view shares memory with $this (rtn->base = $this, refcount bumped)
 * and is registered in the global buffer by ndarray_init_new_object, which
 * also routes 0-D results through NDArray_ScalarToZval and frees them — so
 * no buffer slot is leaked per iteration.
 *
 * @return NDArray|int|float|string|null Per the dtype/rank rules above.
 */
PHP_METHOD(NDArray, current) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);

    /* Guard mirrors PHP's standard Iterator: when the cursor is past the end
       (or the source has no axis 0), current() yields null instead of reading
       off the buffer. PHP foreach normally checks valid() first, but a hand-
       driven loop might not — this keeps that case safe. */
    if (NDArrayIteratorPHP_ISDONE(ndarray)) {
        RETURN_NULL();
    }

    NDArray* result = NDArrayIteratorPHP_GET(ndarray);
    if (result == NULL) {
        RETURN_NULL();
    }
    /* ndarray_init_new_object handles both wrappings: ndim > 0 registers the
       view in the global buffer and exposes it as an NDArray PHP object;
       ndim == 0 routes through NDArray_ScalarToZval (dtype-aware) and frees
       the temporary view. Calling add_to_buffer() here would register a slot
       that the ndim==0 path then frees via NDArray_FREE without clearing,
       leaking one buffer entry per iteration on 1-D foreach. */
    ndarray_init_new_object(result, return_value);
}

/**
 * @brief NDArray::key() — Iterator key (axis-0 index at the cursor).
 *
 * Returns 0 for 0-D source or an NDArray whose php_iterator was never
 * installed (the scalar factory paths skip NDArrayIterator_INIT for ndim==0
 * results). PHP foreach pairs this with valid() === false on those inputs
 * so the loop body never observes the placeholder zero.
 *
 * @return int Current axis-0 index, or 0 when no axis 0 exists.
 */
PHP_METHOD(NDArray, key) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    if (NDArray_NDIM(ndarray) == 0 || ndarray->php_iterator == NULL) {
        RETURN_LONG(0);
    }
    RETURN_LONG(ndarray->php_iterator->currentIndex);
}

/**
 * @brief NDArray::next() — Advance the PHP iterator by one axis-0 step.
 *
 * @return void
 */
PHP_METHOD(NDArray, next) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArrayIteratorPHP_NEXT(ndarray);
}

/**
 * @brief NDArray::rewind() — Reset the PHP iterator to the first element.
 *
 * @return void
 */
PHP_METHOD(NDArray, rewind) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArrayIteratorPHP_REWIND(ndarray);
}

/**
 * @brief Coerce a PHP offset zval into a non-negative axis-0 index.
 *
 * Accepts IS_LONG and finite IS_DOUBLE offsets (the only forms PHP's array-
 * access machinery passes through with a usable numeric value); booleans,
 * strings, arrays, objects, null, and non-finite doubles (NaN, +/-Inf) are
 * all rejected. The coerced long is written back to @p out_index on success.
 *
 * Rejecting NaN / Inf explicitly avoids the undefined behavior of casting
 * them to a C signed integer — without this guard the resulting "negative"
 * value would trigger the wrong error path in the caller's bounds check.
 *
 * @param[in]  offset    PHP zval representing the requested axis-0 index.
 * @param[out] out_index Receives the long-coerced index on success.
 * @return 1 on success; 0 if @p offset is not an integer-coercible value.
 */
static int ndarray_offset_to_long(const zval *offset, zend_long *out_index) {
    if (Z_TYPE_P(offset) == IS_LONG) {
        *out_index = Z_LVAL_P(offset);
        return 1;
    }
    if (Z_TYPE_P(offset) == IS_DOUBLE) {
        double d = Z_DVAL_P(offset);
        /* (zend_long) NaN / Inf is undefined behavior — bail out before the
           cast. ZEND_DOUBLE_FITS_LONG additionally rejects values whose
           magnitude would overflow a signed long, in a platform-correct way
           (the macro definition flips at the ZEND_LONG_MAX boundary). */
        if (!zend_finite(d) || !ZEND_DOUBLE_FITS_LONG(d)) {
            return 0;
        }
        *out_index = (zend_long) d;
        return 1;
    }
    return 0;
}

/**
 * @brief Borrow the axis-0 sub-view of @p ndarray at @p offset, with bounds.
 *
 * Validates the source is at least 1-D, normalises @p offset through
 * ndarray_offset_to_long(), bounds-checks against shape[0], and produces the
 * borrowed view via NDArrayIterator_GET. The iterator cursor is restored to 0
 * before returning so subsequent foreach passes start cleanly. On any failure
 * a Zend Error is thrown and NULL is returned — callers must check first and
 * fall through without producing a return value.
 *
 * Lifecycle of the returned view:
 *   - rtn->base aliases @p ndarray and rtn ADDREFs it
 *   - rtn->data points into @p ndarray's buffer (no copy)
 *   - the caller owns rtn and must release it with NDArray_FREE() — that DECREFs
 *     the source and tears down the view metadata without touching the buffer
 *
 * @param[in,out] ndarray Source NDArray (must be non-NULL and ndim > 0).
 * @param[in]     offset  PHP zval representing the requested axis-0 index.
 * @return Borrowed view NDArray (caller frees), or NULL after throwing.
 */
static NDArray *ndarray_axis0_view_or_throw(NDArray *ndarray, zval *offset) {
    /* 0-D scalars have no axis 0 to index, and their iterator was never
       installed (scalar factories skip NDArrayIterator_INIT) — without this
       guard, accessing `$scalar[0]` would dereference NULL. */
    if (NDArray_NDIM(ndarray) == 0) {
        zend_throw_error(NULL, "Cannot index a 0-D NDArray (no axis 0)");
        return NULL;
    }
    zend_long index;
    if (!ndarray_offset_to_long(offset, &index)) {
        zend_throw_error(NULL, "Invalid offset");
        return NULL;
    }
    if (index < 0) {
        zend_throw_error(NULL, "Negative indexes are not implemented.");
        return NULL;
    }
    if (index > NDArray_SHAPE(ndarray)[0] - 1) {
        zend_throw_error(NULL, "Index out of bounds");
        return NULL;
    }
    ndarray->iterator->currentIndex = (int) index;
    NDArray *rtn = NDArrayIterator_GET(ndarray);
    NDArrayIterator_REWIND(ndarray);
    return rtn;
}

/**
 * @brief NDArray::offsetExists() — implements `isset($a[$offset])`.
 *
 * Returns true when @p offset is an integer-coercible value within
 * `[0, shape[0])`. All other inputs — non-numeric offsets, negative indices,
 * out-of-range indices, and 0-D source arrays — silently yield false to match
 * PHP's standard ArrayAccess convention (isset() must never throw).
 *
 * Device- and dtype-independent: reads only cached shape metadata, so no
 * buffer or device transfer is ever issued.
 *
 * @return bool true if `$a[$offset]` would resolve to a valid element.
 */
PHP_METHOD(NDArray, offsetExists) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *offset;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(offset)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray *ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    /* 0-D source has no axis 0 — every offset is out of range. Reading
       NDArray_SHAPE(ndarray)[0] on a 0-D array is undefined memory access. */
    if (NDArray_NDIM(ndarray) == 0) {
        RETURN_BOOL(0);
    }
    zend_long index;
    if (!ndarray_offset_to_long(offset, &index)) {
        RETURN_BOOL(0);
    }
    if (index < 0 || index > NDArray_SHAPE(ndarray)[0] - 1) {
        RETURN_BOOL(0);
    }
    RETURN_BOOL(1);
}

/**
 * @brief NDArray::offsetGet() — implements `$a[$offset]`.
 *
 * Returns the axis-0 element at @p offset. For N-D source (N >= 2) the result
 * is a fresh NDArray view of rank (N-1) that aliases the source buffer (no
 * copy, no device transfer). For 1-D source the result is a dtype-correct
 * PHP scalar produced by NDArray_ScalarToZval:
 *   - `string` for `float128` and `uint64`
 *   - `int`    for `int8..int64` and `uint8..uint32`
 *   - `float`  for `float4..float64`
 *
 * Works identically on CPU and GPU sources: the GPU read path is a single
 * cudaMemcpy of one element, issued lazily by the scalar conversion. No
 * buffer-slot leak occurs because ndarray_init_new_object routes 0-D results
 * through NDArray_ScalarToZval + NDArray_FREE (which DECREFs the source and
 * tears down view metadata without freeing the aliased buffer).
 *
 * @return NDArray|int|float|string Sub-view or dtype-correct scalar.
 * @throws Error When @p offset is non-integer, negative, out of range, or the
 *               source array is 0-D.
 */
PHP_METHOD(NDArray, offsetGet) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *offset;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(offset)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray *ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArray *rtn = ndarray_axis0_view_or_throw(ndarray, offset);
    if (rtn == NULL) {
        return;
    }
    /* ndarray_init_new_object handles both wrappings: ndim > 0 registers the
       view in the global buffer and exposes it as an NDArray PHP object;
       ndim == 0 routes through NDArray_ScalarToZval (dtype-aware) and frees
       the temporary view — DECREFing the source and tearing down the view's
       metadata without freeing the aliased buffer. */
    ndarray_init_new_object(rtn, return_value);
}

/**
 * @brief Dtype-preserving scalar broadcast into every element of @p slice.
 *
 * Writes a single PHP scalar value (long / double / string / bool) into every
 * element of @p slice using the dtype-aware ndarray_set_from_* hooks.
 *
 * Why this exists: ZVAL_TO_NDARRAY(IS_LONG) routes through
 * NDArray_CreateFromLongScalar which casts to float32 (≈ 7 decimal digits).
 * For int64 / uint64 / float128 targets that's a precision-destroying
 * round-trip — PHP_INT_MAX (9.22e18) overflows the float32 mantissa and
 * comes back as PHP_INT_MIN. This path stays in the target dtype the whole
 * way.
 *
 * On GPU targets the bytes are staged in a host-side typed buffer and pushed
 * with a single NDArray_TypedH2D() (which handles the fp128 → double-double
 * conversion). The temp buffer is freed on every exit path.
 *
 * @param[in,out] slice Destination view (any device, any dtype).
 * @param[in]     value PHP zval holding the source scalar.
 * @return 1 on success; 0 if @p value's type is not a PHP scalar (caller
 *         should fall through to the NDArray_Overwrite path).
 */
static int ndarray_fill_from_php_scalar(NDArray *slice, zval *value) {
    const char *dtype = NDArray_TYPE(slice);
    long  n     = NDArray_NUMELEMENTS(slice);
    int   elsize = NDArray_ELSIZE(slice);
    if (n <= 0) return 1;

    char *target_data;
    char *gpu_tmp = NULL;
    if (NDArray_DEVICE(slice) == NDARRAY_DEVICE_GPU) {
        gpu_tmp = emalloc((size_t)n * (size_t)elsize);
        target_data = gpu_tmp;
    } else {
        target_data = (char *)NDArray_DATA(slice);
    }

    int is_int64_like = (!strcmp(dtype, "int64") || !strcmp(dtype, "uint64"));

    if (Z_TYPE_P(value) == IS_STRING) {
        const char *str = Z_STRVAL_P(value);
        for (long i = 0; i < n; i++) {
            ndarray_set_from_string(dtype, target_data, (size_t)i, str);
        }
    } else if (Z_TYPE_P(value) == IS_LONG) {
        zend_long lv = Z_LVAL_P(value);
        if (is_int64_like) {
            /* Avoid double-rounding of long values for int64/uint64 dtypes. */
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%lld", (long long)lv);
            for (long i = 0; i < n; i++) {
                ndarray_set_from_string(dtype, target_data, (size_t)i, tmp);
            }
        } else {
            for (long i = 0; i < n; i++) {
                ndarray_set_from_double(dtype, target_data, (size_t)i, (double)lv);
            }
        }
    } else if (Z_TYPE_P(value) == IS_DOUBLE) {
        double dv = Z_DVAL_P(value);
        for (long i = 0; i < n; i++) {
            ndarray_set_from_double(dtype, target_data, (size_t)i, dv);
        }
    } else if (Z_TYPE_P(value) == IS_TRUE) {
        for (long i = 0; i < n; i++) {
            ndarray_set_from_double(dtype, target_data, (size_t)i, 1.0);
        }
    } else if (Z_TYPE_P(value) == IS_FALSE) {
        for (long i = 0; i < n; i++) {
            ndarray_set_from_double(dtype, target_data, (size_t)i, 0.0);
        }
    } else {
        if (gpu_tmp) efree(gpu_tmp);
        return 0; /* signal: caller should fall through */
    }

#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(slice) == NDARRAY_DEVICE_GPU) {
        /* fp128 on GPU is stored as double-double — convert during transfer. */
        NDArray_TypedH2D((char *)NDArray_DATA(slice), gpu_tmp, n, dtype);
        efree(gpu_tmp);
    }
#endif
    return 1;
}

/**
 * @brief NDArray::offsetSet() — implements `$a[$offset] = $value`.
 *
 * Writes @p value into the axis-0 slice at @p offset, preserving the array's
 * dtype, shape, and device. Two paths share the same borrowed view of axis 0:
 *
 *   1. PHP scalar (long / double / string / bool) → @ref ndarray_fill_from_php_scalar
 *      broadcasts the value across every element of the slice while keeping
 *      end-to-end byte fidelity for int64 / uint64 / float128. On GPU the
 *      bytes are staged in a host-side typed buffer and pushed with a single
 *      cudaMemcpy.
 *   2. PHP array or NDArray → wrapped with ZVAL_TO_NDARRAY() and copied with
 *      NDArray_Overwrite(), which fast-paths same-dtype same-device traffic
 *      via memcpy / vmemcpyd2d and otherwise casts element-wise through
 *      double on CPU.
 *
 * Memory: the borrowed view ADDREFs the source on creation and is released
 * with NDArray_FREE on every exit path, so repeated assignments do not leak
 * buffer slots or VRAM. Any temporary NDArray produced from a PHP array /
 * scalar input is freed via CHECK_INPUT_AND_FREE.
 *
 * @throws Error On 0-D source, non-integer / negative / out-of-range offset,
 *               or a value type that cannot be coerced to an NDArray.
 */
PHP_METHOD(NDArray, offsetSet) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *offset;
    zval *value;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(offset)
        Z_PARAM_ZVAL(value)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray *ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArray *rtn = ndarray_axis0_view_or_throw(ndarray, offset);
    if (rtn == NULL) {
        return;
    }

    /* Scalar/string fast path stays in the target's dtype end-to-end. */
    if (Z_TYPE_P(value) == IS_LONG   || Z_TYPE_P(value) == IS_DOUBLE ||
        Z_TYPE_P(value) == IS_STRING || Z_TYPE_P(value) == IS_TRUE   ||
        Z_TYPE_P(value) == IS_FALSE) {
        if (ndarray_fill_from_php_scalar(rtn, value)) {
            NDArray_FREE(rtn);
            return;
        }
    }

    /* Fallback: array / NDArray source — wrap and let NDArray_Overwrite
       handle byte-copy or cast through double. ZVAL_TO_NDARRAY throws on
       unsupported value types and returns NULL, in which case we release
       the borrowed view and propagate the pending exception. */
    NDArray *nd_value = ZVAL_TO_NDARRAY(value);
    if (nd_value == NULL) {
        NDArray_FREE(rtn);
        return;
    }
    NDArray_Overwrite(rtn, nd_value);
    NDArray_FREE(rtn);
    CHECK_INPUT_AND_FREE(value, nd_value);
}

/* Wire format emitted by __serialize:
 *   ['__ndarray__' => 1, 'dtype' => 'float128', 'shape' => [3], 'data' => [...]]
 *
 * The dtype field lets __unserialize reconstruct the original array with full
 * precision — without it, the receiver defaults to float32 and silently
 * round-trips float128/uint64/int64 through double, dropping bits.
 *
 * Backward compatibility: __unserialize first checks for the '__ndarray__'
 * marker. If absent, it falls back to treating the payload as a plain PHP
 * array and constructing a float32 NDArray (the historical lossy behaviour).
 */
PHP_METHOD(NDArray, __serialize) {
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

    array_init(return_value);

    add_assoc_long(return_value, "__ndarray__", 1);
    add_assoc_string(return_value, "dtype", (char *)NDArray_TYPE(array));

    zval shape_arr;
    array_init(&shape_arr);
    for (int i = 0; i < NDArray_NDIM(array); i++) {
        add_next_index_long(&shape_arr, (zend_long)NDArray_SHAPE(array)[i]);
    }
    add_assoc_zval(return_value, "shape", &shape_arr);

    zval data_zv;
    if (NDArray_NDIM(array) == 0) {
        NDArray_ScalarToZval(array, &data_zv);
    } else {
        data_zv = NDArray_ToPHPArray(array);
    }
    add_assoc_zval(return_value, "data", &data_zv);
}

PHP_METHOD(NDArray, __unserialize) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(data)
    ZEND_PARSE_PARAMETERS_END();

    /* New format: {'__ndarray__': 1, 'dtype': str, 'shape': [...], 'data': mixed} */
    if (Z_TYPE_P(data) == IS_ARRAY) {
        HashTable *ht = Z_ARRVAL_P(data);
        zval *marker = zend_hash_str_find(ht, "__ndarray__", sizeof("__ndarray__") - 1);
        zval *dtype  = zend_hash_str_find(ht, "dtype",       sizeof("dtype") - 1);
        zval *payload = zend_hash_str_find(ht, "data",       sizeof("data") - 1);

        if (marker != NULL && dtype != NULL && payload != NULL &&
            Z_TYPE_P(dtype) == IS_STRING) {
            /* Canonicalise: descriptor->type is stored by reference, so we
               must use the static NDARRAY_TYPE_* pointer rather than the
               soon-to-be-freed Z_STRVAL_P buffer. */
            const char *canonical = type_canonicalize(Z_STRVAL_P(dtype));
            if (canonical == NULL) {
                zend_throw_error(NULL,
                    "Unknown dtype '%s' in serialised NDArray payload",
                    Z_STRVAL_P(dtype));
                return;
            }
            NDArray *nda = NDArrayFactory_createFromZval(payload, canonical);
            if (nda == NULL) {
                if (!EG(exception)) {
                    zend_throw_error(NULL,
                        "Failed to reconstruct NDArray from unserialised payload");
                }
                return;
            }
            /* NDArrayFactory_createFromZval already called add_to_buffer. */
            ZVAL_LONG(OBJ_PROP_NUM(obj, 0), NDArray_UUID(nda));
            return;
        }
    }

    /* Legacy format: plain PHP array, default to float32 (lossy for non-fp32 dtypes). */
    NDArray *nda = ZVAL_TO_NDARRAY(data);
    if (nda == NULL) {
        return;
    }
    add_to_buffer(nda);
    ZVAL_LONG(OBJ_PROP_NUM(obj, 0), NDArray_UUID(nda));
}


/**
 * @brief NDArray::offsetUnset() — implements `unset($a[$offset])` (rejected).
 *
 * NDArray buffers are fixed-shape, fixed-dtype typed arrays — there is no
 * tombstone state that would let an element be marked "absent" without
 * disturbing the surrounding strides or shape. Per NumPy semantics, deleting
 * an individual element is not a meaningful operation; callers wanting to
 * zero a slice should assign `0` (or the dtype-appropriate string) instead.
 * The @p offset is accepted as @c mixed only because the ArrayAccess
 * contract requires the signature.
 *
 * @throws Error Unconditionally.
 */
PHP_METHOD(NDArray, offsetUnset) {
    zval *offset;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(offset)
    ZEND_PARSE_PARAMETERS_END();
    (void) offset;
    zend_throw_error(NULL, "Cannot unset values of NDArrays");
}

/**
 * @brief NDArray::valid() — Iterator validity at the current cursor.
 *
 * Returns false once the cursor has reached or passed axis-0 length, and
 * unconditionally false for 0-D source (no axis 0 to enumerate). This is the
 * gate PHP foreach uses before each current()/key().
 *
 * @return bool true if current()/key() would return a meaningful value.
 */
PHP_METHOD(NDArray, valid) {
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    RETURN_BOOL(!NDArrayIteratorPHP_ISDONE(ndarray));
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
    /* Private + final: NumPower is a static factory only. `new NumPower()`
       must fail with "Call to private NumPower::__construct() from global
       scope" — there is no per-instance state for a constructor to set up.
       Combined with ZEND_ACC_FINAL on the class entry, this also blocks any
       subclass from re-exposing a public constructor. */
    ZEND_ME(NumPower, __construct, arginfo_NumPower_construct, ZEND_ACC_PRIVATE | ZEND_ACC_FINAL)

    // EXTREMA
    ZEND_ME(NumPower, min, arginfo_ndarray_min, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, max, arginfo_ndarray_max, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, maximum, arginfo_ndarray_maximum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, minimum, arginfo_ndarray_minimum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, argmax, arginfo_ndarray_argmax, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, argmin, arginfo_ndarray_argmin, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    // MANIPULATION
    ZEND_ME(NumPower, reshape, arginfo_reshape, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NumPower, slice, arginfo_numpower_slice, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
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
    ZEND_ME(NumPower, rc, arginfo_ndarray_rc, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
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
    ZEND_ME(NumPower, syncDevice, arginfo_ndarray_devicesync, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

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

/**
 * @brief Register the NumPower factory class.
 *
 * NumPower is a static factory: every operation it exposes is `ZEND_ACC_STATIC`
 * (zeros, ones, array, matmul, …). It deliberately does NOT implement
 * Iterator / Countable / ArrayAccess — those interfaces would force PHP to
 * mark the class abstract (the methods are not provided), forbidding
 * `new NumPower()` entirely. The empty `__construct` is retained for binary
 * compatibility but stores no per-instance state.
 *
 * The unused `class_entry_Iterator / Countable / ArrayAccess` parameters are
 * kept to preserve the call site in MINIT and the signature shared with
 * register_class_NDArray.
 *
 * @return Registered class entry.
 */
static zend_class_entry *register_class_NumPower(zend_class_entry *class_entry_Iterator, zend_class_entry *class_entry_Countable, zend_class_entry *class_entry_ArrayAccess) {
    (void)class_entry_Iterator;
    (void)class_entry_Countable;
    (void)class_entry_ArrayAccess;
    zend_class_entry ce, *class_entry;
    INIT_CLASS_ENTRY(ce, "NumPower", class_NumPower_methods);
    ndarray_objects_init(&ce);
    ce.create_object = ndarray_create_object;
    class_entry = zend_register_internal_class(&ce);
    /* Static-factory class: subclassing makes no sense and would inherit a
       no-op constructor plus 130 unrelated statics. Mark final to match the
       stub.php declaration and forbid `class X extends NumPower`. */
    class_entry->ce_flags |= ZEND_ACC_FINAL;

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
    buffer_free();
#ifdef ZTS
    if (MAIN_MEM_STACK.lock) {
        tsrm_mutex_free(MAIN_MEM_STACK.lock);
        MAIN_MEM_STACK.lock = NULL;
    }
#endif
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
