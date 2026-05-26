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

/* ── Reduction-as-NDArray (preserves dtype) ─────────────────────────────────
   Return the no-axis reduction as a 0-D NDArray whose dtype matches @p a's.
   Routes through `NDArray_Reduce_*` for the actual computation and stores
   the resulting double into an output buffer of the correct dtype via
   `ndarray_set_from_double`. Used by the sum/prod PHP entry points so the
   caller can dispatch to `NDArray_ScalarToZval` and return the dtype-aware
   PHP scalar (`string` for `float128`/`uint64`, `int`/`float` otherwise). */
NDArray *NDArray_Reduce_Sum_AsNDArray(NDArray *a);
NDArray *NDArray_Reduce_Prod_AsNDArray(NDArray *a);

/* ── Axis-based reductions ─────────────────────────────────────────────────
   Reduce along a single axis @p axis (must be in range [0, ndim)). The
   output dtype matches @p a's, and the device is preserved (CPU input
   → CPU output; GPU input → GPU output, no host staging). The output
   shape is @p a's shape with @p axis removed; for a 1-D input the output
   is a 0-D NDArray. */
enum ndarray_reduce_axis_op {
    ND_AXIS_RED_SUM,
    ND_AXIS_RED_PROD
};
NDArray *NDArray_Reduce_Axis(NDArray *a, int axis,
                              enum ndarray_reduce_axis_op op);
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

/* Native CPU binary op for every integer dtype (`int8`..`int64`,
   `uint8`..`uint64`). Avoids the float round-trip used by the generic
   arithmetic dispatcher so:
    - PyTorch's modular wrap-around semantics survive for every dtype,
    - `int64` / `uint64` precision past 2^53 is preserved,
    - the result matches the GPU `cuda_<op>_<tag>` kernels bit-for-bit
      (the prior float64 → narrow-int double-cast diverged between CPU
      and GPU once intermediates spilled past 2^53).
   Both operands must be on CPU and share the dtype. One operand may be
   a 0-D scalar. opcode is ZEND_ADD/SUB/MUL/MOD/POW; ZEND_DIV is promoted
   to float upstream by ndarray_div_promote and never reaches here. */
NDArray* NDArray_TypedBinOp_CPU_Int(int opcode, NDArray* a, NDArray* b);

/* Backward-compatible alias of NDArray_TypedBinOp_CPU_Int — retained
   so external callers built against the prior int64-only entry point
   keep linking. Forwards to the dtype-generic helper above. */
NDArray* NDArray_TypedBinOp_CPU_Int64(int opcode, NDArray* a, NDArray* b);

NDArray* NDArray_Add(NDArray* a, NDArray* b);

#endif //PHPSCI_NDARRAY_ARITHMETICS_H
