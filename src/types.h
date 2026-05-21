#ifndef PHPSCI_NDARRAY_TYPES_H
#define PHPSCI_NDARRAY_TYPES_H

#include "ndarray_types.h"

/* All supported dtype strings */
static const char* NDARRAY_TYPE_FLOAT4   = "float4";
static const char* NDARRAY_TYPE_FLOAT8   = "float8";
static const char* NDARRAY_TYPE_FLOAT16  = "float16";
static const char* NDARRAY_TYPE_FLOAT32  = "float32";
static const char* NDARRAY_TYPE_FLOAT64  = "float64";
static const char* NDARRAY_TYPE_FLOAT128 = "float128";
static const char* NDARRAY_TYPE_INT8     = "int8";
static const char* NDARRAY_TYPE_UINT8    = "uint8";
static const char* NDARRAY_TYPE_INT16    = "int16";
static const char* NDARRAY_TYPE_UINT16   = "uint16";
static const char* NDARRAY_TYPE_INT32    = "int32";
static const char* NDARRAY_TYPE_UINT32   = "uint32";
static const char* NDARRAY_TYPE_INT64    = "int64";
static const char* NDARRAY_TYPE_UINT64   = "uint64";

/* Returns element size in bytes, or 0 for unknown types */
int get_type_size(const char *type);

/* Returns 1 if type_a and type_b are the same dtype string */
int is_type(const char *type_a, const char *type_b);

/* Returns 1 if the type needs string-based I/O from PHP (no native PHP representation) */
int type_needs_string_io(const char *type);

/* Returns 1 if the type is one of the validated dtype strings */
int type_is_valid(const char *type);

/* Returns the canonical pointer (static program-lifetime constant) for the
   given dtype name, or NULL if unrecognised. Use this whenever the source
   buffer of the type string may be freed (e.g. PHP strings during
   __unserialize) — NDArrayDescriptor stores the pointer by reference. */
const char *type_canonicalize(const char *type);

/* PyTorch-compatible type promotion: returns the higher-ranked dtype of a and b */
const char *promote_dtype(const char *a, const char *b);

/* Returns the dtype to use for arithmetic computation (e.g., float16 → float32) */
const char *compute_dtype_for_arithmetic(const char *result_dtype);

#endif /* PHPSCI_NDARRAY_TYPES_H */
