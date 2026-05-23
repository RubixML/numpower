#include <string.h>
#include <php.h>
#include "Zend/zend_alloc.h"
#include "iterators.h"
#include "ndarray.h"
#include "initializers.h"

/**
 * @brief Test whether @p iter has advanced past the last axis-0 element.
 *
 * 0-D arrays carry no axis 0, so they are treated as already done — this lets
 * PHP foreach yield no values and never call current()/key() (which would
 * otherwise dereference a non-existent shape[0]).
 *
 * @param[in] array Source NDArray.
 * @param[in] iter  Iterator state to consult.
 * @return 1 if @p iter has reached or passed the end, 0 otherwise.
 */
static int
iterator_is_done(const NDArray* array, const NDArrayIterator* iter) {
    /* 0-D or missing iterator state -> nothing to walk, done up-front. */
    if (array->ndim == 0 || iter == NULL) {
        return 1;
    }
    return iter->currentIndex >= NDArray_SHAPE(array)[0] ? 1 : 0;
}

/**
 * @brief Produce a borrowed-data sub-view at @p iter's current axis-0 index.
 *
 * The returned NDArray aliases @p array's data buffer (rtn->data points into
 * it, rtn->base holds @p array) and bumps @p array's refcount so the view
 * stays valid while in use. For an N-D input the view is (N-1)-D; for a 1-D
 * input the view is 0-D so the caller (NDArray_ScalarToZval) can read a
 * single dtype-correct scalar. 0-D inputs have no element to return, so this
 * function returns NULL and callers must guard with iterator_is_done() first.
 *
 * @param[in] array Source NDArray.
 * @param[in] iter  Iterator state describing the offset along axis 0.
 * @return Newly-allocated view NDArray, or NULL when @p array is 0-D.
 */
static NDArray*
iterator_view_at(NDArray* array, const NDArrayIterator* iter) {
    if (array->ndim == 0 || iter == NULL) {
        return NULL;
    }

    int output_ndim = array->ndim - 1;
    /* Always allocate at least one int — emalloc(0) is technically valid but
       the descendant Create_NDArray reads shape[0] before the ndim==0 guard
       (where it overwrites to 1), and reading a zero-byte allocation trips
       Valgrind / ASan. The slot is harmlessly unused for output_ndim == 0. */
    int shape_slots = output_ndim > 0 ? output_ndim : 1;
    int* output_shape = emalloc(sizeof(int) * (size_t)shape_slots);
    if (output_ndim > 0) {
        memcpy(output_shape, NDArray_SHAPE(array) + 1,
               sizeof(int) * (size_t)output_ndim);
    } else {
        output_shape[0] = 0;
    }

    NDArray_ADDREF(array);
    NDArray* rtn = Create_NDArray(output_shape, output_ndim,
                                  NDArray_TYPE(array), NDArray_DEVICE(array));
    rtn->device = NDArray_DEVICE(array);
    rtn->data = array->data
                + ((size_t)iter->currentIndex
                   * (size_t)NDArray_STRIDES(array)[0]);
    rtn->base = array;
    return rtn;
}

/**
 * @brief Internal-iterator ISDONE check. @see iterator_is_done.
 *
 * @param[in] array Source NDArray.
 * @return 1 if the iterator has reached or passed the end, 0 otherwise.
 */
int
NDArrayIterator_ISDONE(NDArray* array) {
    return iterator_is_done(array, array->iterator);
}

/**
 * @brief Advance the internal iterator by one axis-0 step.
 *
 * No-op on 0-D arrays (no axis 0 to walk) and on arrays whose iterator state
 * was never installed — the 0-D scalar factory paths skip
 * NDArrayIterator_INIT, so callers driving foreach on a freshly-built scalar
 * would otherwise dereference an uninitialized pointer.
 *
 * @param[in,out] array NDArray whose internal iterator advances.
 */
void
NDArrayIterator_NEXT(NDArray* array) {
    if (array->ndim == 0 || array->iterator == NULL) {
        return;
    }
    array->iterator->currentIndex++;
}

/**
 * @brief Reset the internal iterator to the first axis-0 element.
 *
 * Same 0-D / NULL guard as NDArrayIterator_NEXT.
 *
 * @param[in,out] array NDArray whose internal iterator is reset.
 */
void
NDArrayIterator_REWIND(NDArray* array) {
    if (array->ndim == 0 || array->iterator == NULL) {
        return;
    }
    array->iterator->currentIndex = 0;
}

/**
 * @brief PHP-iterator ISDONE check (drives valid()). @see iterator_is_done.
 *
 * @param[in] array Source NDArray.
 * @return 1 if the PHP iterator has reached or passed the end, 0 otherwise.
 */
int
NDArrayIteratorPHP_ISDONE(NDArray* array) {
    return iterator_is_done(array, array->php_iterator);
}

/**
 * @brief Advance the PHP iterator by one axis-0 step (drives next()).
 *
 * No-op on 0-D arrays (no axis 0 to walk) and on arrays whose php_iterator
 * was never installed — the 0-D scalar factory paths skip
 * NDArrayIterator_INIT, so a foreach over a freshly-built scalar must not
 * dereference an uninitialized pointer.
 *
 * @param[in,out] array NDArray whose PHP iterator advances.
 */
void
NDArrayIteratorPHP_NEXT(NDArray* array) {
    if (array->ndim == 0 || array->php_iterator == NULL) {
        return;
    }
    array->php_iterator->currentIndex++;
}

/**
 * @brief Reset the PHP iterator to the first axis-0 element (drives rewind()).
 *
 * Same 0-D / NULL guard as NDArrayIteratorPHP_NEXT.
 *
 * @param[in,out] array NDArray whose PHP iterator is reset.
 */
void
NDArrayIteratorPHP_REWIND(NDArray* array) {
    if (array->ndim == 0 || array->php_iterator == NULL) {
        return;
    }
    array->php_iterator->currentIndex = 0;
}

/**
 * @brief Borrow the axis-0 sub-view at the PHP iterator's current index.
 *
 * Used by current(). Callers must check NDArrayIteratorPHP_ISDONE() first —
 * 0-D source arrays and out-of-range indices return NULL here rather than
 * dereferencing missing shape metadata or reading off the buffer end.
 *
 * @param[in] array Source NDArray.
 * @return Newly-allocated view NDArray (caller owns), or NULL for 0-D inputs.
 */
NDArray*
NDArrayIteratorPHP_GET(NDArray* array) {
    return iterator_view_at(array, array->php_iterator);
}

/**
 * @brief Allocate and zero both iterator state structs on @p array.
 *
 * Called once per NDArray at construction time. The two iterators are
 * independent — the PHP one is driven by foreach / current() / next(), while
 * the internal one is used by C-level traversals (broadcast, axis reductions,
 * etc.) — so a foreach in user code does not perturb internal algorithms.
 *
 * @param[in,out] array NDArray to initialize.
 */
void
NDArrayIterator_INIT(NDArray* array) {
    NDArrayIterator* iterator = (NDArrayIterator*)emalloc(sizeof(NDArrayIterator));
    NDArrayIterator* php_iterator = (NDArrayIterator*)emalloc(sizeof(NDArrayIterator));
    iterator->currentIndex = 0;
    php_iterator->currentIndex = 0;
    array->iterator = iterator;
    array->php_iterator = php_iterator;
}

/**
 * @brief Borrow the axis-0 sub-view at the internal iterator's current index.
 *
 * Counterpart of NDArrayIteratorPHP_GET for C-internal traversals. The same
 * NULL-on-0-D contract applies; callers must check NDArrayIterator_ISDONE().
 *
 * @param[in] array Source NDArray.
 * @return Newly-allocated view NDArray (caller owns), or NULL for 0-D inputs.
 */
NDArray*
NDArrayIterator_GET(NDArray* array) {
    return iterator_view_at(array, array->iterator);
}

/**
 * @brief Release the iterator state structs owned by @p array.
 *
 * Called from NDArray_FREE on the final refcount drop, so the matching
 * emalloc()s in NDArrayIterator_INIT are returned to the Zend MM.
 *
 * @param[in,out] array NDArray whose iterators are freed.
 */
void
NDArrayIterator_FREE(NDArray* array) {
    if (array->iterator != NULL) {
        efree(array->iterator);
        array->iterator = NULL;
    }
    if (array->php_iterator != NULL) {
        efree(array->php_iterator);
        array->php_iterator = NULL;
    }
}

NDArrayIter*
NDArray_NewElementWiseIter(NDArray *target) {
    NDArrayIter *it;
    int i, nd;
    NDArray *ao = target;

    it = emalloc(sizeof(NDArrayIter));
    if (it == NULL) {
        return NULL;
    }

    nd = NDArray_NDIM(ao);
    it->contiguous = 1;
    if (NDArray_checkFlags(target, NDARRAY_ARRAY_F_CONTIGUOUS)) {
        it->contiguous = 0;
    }
    it->ao = ao;
    it->size = NDArray_NUMELEMENTS(ao);
    it->nd_m1 = nd - 1;
    it->factors[nd-1] = 1;
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = NDArray_SHAPE(it->ao)[i] - 1;
        it->strides[i] = NDArray_STRIDES(it->ao)[i];
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * it->ao->dimensions[nd-i];
        }
    }
    NDArray_ITER_RESET(it);
    return it;
}

int
NDArray_PrepareTwoRawArrayIter(int ndim, int const *shape,
                               char *dataA, int const *stridesA,
                               char *dataB, int const *stridesB,
                               int *out_ndim, int *out_shape,
                               char **out_dataA, int *out_stridesA,
                               char **out_dataB, int *out_stridesB)
{
    NDArrayStrideSortItem strideperm[NDARRAY_MAX_DIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        int stride_entryA = stridesA[0], stride_entryB = stridesB[0];
        int shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    NDArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        int stride_entryA = out_stridesA[i];
        int stride_entryB = out_stridesB[i];
        int shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                 out_stridesB[i] * out_shape[i] == out_stridesB[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_ndim = ndim;
    return 0;
}


