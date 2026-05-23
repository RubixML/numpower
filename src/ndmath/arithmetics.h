#ifndef PHPSCI_NDARRAY_ARITHMETICS_H
#define PHPSCI_NDARRAY_ARITHMETICS_H

#include "../ndarray.h"

//FLoats
NDArray* NDArray_Subtract_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Add_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Multiply_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Divide_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Pow_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Mod_Float(NDArray* a, NDArray* b);
float NDArray_Sum_Float(NDArray* a);
float NDArray_Float_Prod(NDArray* a);
float NDArray_Mean_Float(NDArray* a);

/* ── dtype-aware reductions ─────────────────────────────────────────────────
   Read the source buffer with the correct stride for every supported dtype
   and accumulate in double. The legacy NDArray_Sum_Float / NDArray_Float_Prod
   / NDArray_Mean_Float / NDArray_Min / NDArray_Max helpers cast the buffer
   to (float*) regardless of dtype, which returns garbage for any non-float32
   input. These helpers fix that for the no-axis PHP entry points. */
double NDArray_Reduce_Sum(NDArray *a);
double NDArray_Reduce_Prod(NDArray *a);
double NDArray_Reduce_Min(NDArray *a);
double NDArray_Reduce_Max(NDArray *a);
double NDArray_Reduce_Mean(NDArray *a);
float NDArray_Mean_Float_Axis(NDArray* a, NDArray *b);
NDArray* NDArray_Abs(NDArray *nda);
float NDArray_Median_Float(NDArray* a);

//Doubles
NDArray* NDArray_Subtract_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Add_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Pow_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Multiply_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Divide_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Mod_Double(NDArray* a, NDArray* b);
double NDArray_Sum_Double(NDArray* a);

//Float128 (CPU: native __float128 on x86-64 GCC, libquadmath used for pow/mod
// when available; GPU: double-double emulation via dd kernels.)
NDArray* NDArray_Add_Float128(NDArray* a, NDArray* b);
NDArray* NDArray_Subtract_Float128(NDArray* a, NDArray* b);
NDArray* NDArray_Multiply_Float128(NDArray* a, NDArray* b);
NDArray* NDArray_Divide_Float128(NDArray* a, NDArray* b);
NDArray* NDArray_Pow_Float128(NDArray* a, NDArray* b);
NDArray* NDArray_Mod_Float128(NDArray* a, NDArray* b);

/* Typed GPU binary op: both inputs must already be on GPU and of the same
   dtype (one operand may be a 0-dim scalar). Dispatches to the right CUDA
   kernel for every native dtype. Returns a new GPU NDArray. opcode is one
   of the ZEND_ADD/SUB/MUL/DIV/POW/MOD constants. */
NDArray* NDArray_TypedBinOp_GPU(int opcode, NDArray* a, NDArray* b);

NDArray* NDArray_Add(NDArray* a, NDArray* b);

#endif //PHPSCI_NDARRAY_ARITHMETICS_H
