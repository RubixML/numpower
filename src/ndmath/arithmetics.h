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
   to float upstream by ndarray_widen_int_to_float and never reaches here. */
NDArray* NDArray_TypedBinOp_CPU_Int(int opcode, NDArray* a, NDArray* b);

NDArray* NDArray_Add(NDArray* a, NDArray* b);

/* Sentinel "opcode" for the two-argument arctangent (atan2). It rides the
   shared typed binary dispatch (`ndarray_promote_and_op` →
   `NDArray_TypedBinOp_GPU` / the CPU float kernels) so atan2 reuses the same
   device-migration, broadcasting, dtype-promotion, cast-back and leak
   handling as +,-,*,/,**,% instead of duplicating it. The value sits above
   `ZEND_VM_LAST_OPCODE` (210 on PHP 8.5) and well clear of the six real
   arithmetic opcodes (ADD/SUB/MUL/DIV/MOD=1..5, POW=12) the dispatch also
   handles, so it can never collide; it fits the `zend_uchar` opcode slot. */
#define NDARRAY_BINOP_ATAN2 250

/* Element-wise two-argument arctangent atan2(a, b) for the three float
   compute dtypes. `ndarray_promote_and_op` promotes both operands to a
   common float dtype (float32 / float64 / float128) before calling, so
   these never see integer input. CPU-only — GPU residency is handled by
   `NDArray_TypedBinOp_GPU`. Argument order follows NumPy: `arctan2(a, b)`
   == C `atan2(a, b)` (a is the numerator, b the denominator). */
NDArray* NDArray_Arctan2_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Arctan2_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Arctan2_Float128(NDArray* a, NDArray* b);

/* ── Typed unary dispatch ────────────────────────────────────────────────────
   Element-wise unary operations dispatched across every supported dtype
   (`float4`..`float128`, `int8`..`uint64`) for both CPU and GPU residency.
   The dispatcher decides result dtype per op (sqrt/rsqrt/reciprocal/sinc
   promote integer inputs to floating-point per PyTorch widening rules; the
   remaining ops preserve the input dtype), allocates the destination buffer
   on the input's device, and runs the typed kernel without a CPU fallback
   for GPU-resident inputs.

   `clip_min` / `clip_max` are decimal strings parsed losslessly into the
   computed dtype (so `float128`/`uint64` survive end-to-end). For ops other
   than `NDARRAY_UNOP_CLIP` both may be NULL. */
typedef enum {
    NDARRAY_UNOP_ABS = 0,
    NDARRAY_UNOP_NEGATIVE,
    NDARRAY_UNOP_POSITIVE,
    NDARRAY_UNOP_RECIPROCAL,
    NDARRAY_UNOP_SIGN,
    NDARRAY_UNOP_SQRT,
    NDARRAY_UNOP_RSQRT,
    NDARRAY_UNOP_SQUARE,
    NDARRAY_UNOP_CLIP,
    NDARRAY_UNOP_SINC,
    NDARRAY_UNOP_EXP,
    NDARRAY_UNOP_EXP2,
    NDARRAY_UNOP_EXPM1,
    NDARRAY_UNOP_LOG,
    NDARRAY_UNOP_LOG2,
    NDARRAY_UNOP_LOG10,
    NDARRAY_UNOP_LOG1P,
    NDARRAY_UNOP_LOGB,
    /* ── Trig / hyperbolic / angle / rounding block ──────────────────────
       The contiguous range `[SIN, CEIL]` is required by
       `unary_op_is_trig` in src/ndmath/arithmetics.c, which uses a single
       range check (`op >= SIN && op <= CEIL`) to pick `UNARY_TRIG_BODY`
       over `UNARY_FLOAT_BODY`. New ops inserted between `SIN` and `CEIL`
       must also live in `UNARY_TRIG_BODY`'s switch; new non-trig ops
       must NOT land inside this block. */
    /* Trigonometric */
    NDARRAY_UNOP_SIN,
    NDARRAY_UNOP_COS,
    NDARRAY_UNOP_TAN,
    NDARRAY_UNOP_ARCSIN,
    NDARRAY_UNOP_ARCCOS,
    NDARRAY_UNOP_ARCTAN,
    /* Hyperbolic */
    NDARRAY_UNOP_SINH,
    NDARRAY_UNOP_COSH,
    NDARRAY_UNOP_TANH,
    NDARRAY_UNOP_ARCSINH,
    NDARRAY_UNOP_ARCCOSH,
    NDARRAY_UNOP_ARCTANH,
    /* Angle conversion */
    NDARRAY_UNOP_DEGREES,
    NDARRAY_UNOP_RADIANS,
    /* Rounding (preserves dtype for floats; integers pass through) */
    NDARRAY_UNOP_RINT,
    NDARRAY_UNOP_FIX,
    NDARRAY_UNOP_TRUNC,
    NDARRAY_UNOP_FLOOR,
    NDARRAY_UNOP_CEIL
} NDArrayUnaryOp;

/**
 * @brief Dispatch the typed unary op @p op on @p nda.
 *
 * @param[in] op       Operation selector (see `NDArrayUnaryOp`).
 * @param[in] nda      Input NDArray on either device.
 * @param[in] clip_min Decimal string for the lower clamp; NULL unless
 *                     @p op is `NDARRAY_UNOP_CLIP`.
 * @param[in] clip_max Decimal string for the upper clamp; NULL unless
 *                     @p op is `NDARRAY_UNOP_CLIP`.
 * @return Caller-owned result NDArray on the same device as @p nda, or
 *         NULL on error (PHP exception in flight).
 */
NDArray *NDArray_TypedUnaryOp(NDArrayUnaryOp op, NDArray *nda,
                              const char *clip_min, const char *clip_max);

#endif //PHPSCI_NDARRAY_ARITHMETICS_H
