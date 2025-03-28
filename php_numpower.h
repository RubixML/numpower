/* phpsci_ndarray extension for PHP */

#ifndef PHP_NDARRAY_H
# define PHP_NDARRAY_H

#include "config.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_print_r_, 0, 1, IS_VOID, 0)
                ZEND_ARG_TYPE_INFO(0, var, IS_MIXED, 0)
                ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, do_return, boolean, 0, "false")
ZEND_END_ARG_INFO()


ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_current, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_next, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_valid, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_rewind, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_key, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_count, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_offsetget, 0, 1, IS_MIXED, 0)
                ZEND_ARG_TYPE_INFO(0, offset, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_offsetset, 0, 2, IS_VOID, 0)
                ZEND_ARG_TYPE_INFO(0, offset, IS_MIXED, 0)
                ZEND_ARG_TYPE_INFO(0, value, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_offsetunset, 0, 1, IS_VOID, 0)
                ZEND_ARG_TYPE_INFO(0, offset, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_offsetexists, 0, 1, _IS_BOOL, 0)
                ZEND_ARG_TYPE_INFO(0, offset, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_TENTATIVE_RETURN_TYPE_INFO_EX(arginfo_unserialize, 0, 1, IS_VOID, 0)
ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_FUNCTION(print_r_);


PHPAPI zend_class_entry *phpsci_ce_NDArray;
PHPAPI zend_class_entry *phpsci_ce_NumPower;
PHPAPI zend_class_entry *phpsci_ce_ArithmeticOperand;

# define PHP_NDARRAY_VERSION "0.7.0"

# if defined(ZTS) && defined(COMPILE_DL_NDARRAY)
ZEND_TSRMLS_CACHE_EXTERN()
# endif

#endif	/* PHP_NDARRAY_H */
