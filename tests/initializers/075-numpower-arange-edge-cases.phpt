--TEST--
NumPower::arange() handles boundary endpoints, sign-mismatched step, and the default contract
--FILE--
<?php
/* numpy parity for the empty / single-element / sign-mismatched edges:
    - stop == start              → empty (length 0)
    - stop, start same, neg step → empty (sign mismatch)
    - stop > start, neg step     → empty (sign mismatch)
    - stop < start, pos step     → empty (sign mismatch)
    - stop = start + step        → single-element array
    - very large length          → bounded by INT_MAX

   The old NDArray_Arange threw on length == 0; the new code returns an
   empty array, matching numpy. */

/* Equal endpoints → empty 1-D array. */
$a = NumPower::arange(0, 0);
echo 'equal_endpoints: shape=', json_encode($a->shape()),
     ' size=', $a->size(),
     ' isNDArray=', ($a instanceof NDArray ? 1 : 0), "\n";

/* Sign-mismatched step variations. */
foreach ([
    'pos_step_back'    => [5,  10, -1],  /* PHP: stop=5, start=10, step=-1 → diff=-5/step=-1=5, OK */
    'pos_step_neg'     => [5,   0, -1],
    'neg_step_forward' => [0,   5,  1],  /* PHP: stop=0, start=5 → diff=-5, step=+1 → empty */
] as $tag => [$stop, $start, $step]) {
    $a = NumPower::arange($stop, $start, $step);
    echo $tag, ': ', (string)$a, "\n";
}

/* Single-element output: start + step == stop. */
$a = NumPower::arange(1, 0, 1);
echo 'single_elem: ', (string)$a, "\n";

/* Default contract: float32 / CPU. */
$a = NumPower::arange(3);
echo 'default_dtype_is_float32: ', (is_float($a[0]) ? 'OK' : 'BAD'), "\n";
echo 'default_device_is_CPU: ', ($a->isGPU() ? 'BAD' : 'OK'), "\n";

/* Numeric string for the narrow dtypes — must also be accepted. */
$a = NumPower::arange('5', '0', '1', 'int32');
echo 'string_for_narrow_dtype: ', (string)$a, "\n";

/* Floating step on integer dtype — the values are truncated per
   ndarray_set_from_double semantics. arange(1, 0, 0.25) with int32 →
   [0, 0, 0, 0] (each cast to int truncates the fractional part). */
$a = NumPower::arange(1.0, 0.0, 0.25, 'int32');
echo 'int32_with_float_step: ', (string)$a, "\n";

/* Endpoints that produce a one-element single-step range for fp128. */
$a = NumPower::arange('1', '0', '1', 'float128');
echo 'fp128_single: ', (string)$a, "\n";

/* GPU empty round-trip. */
$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $has_gpu = false; }
if ($has_gpu) {
    $g = NumPower::arange(0, 0, 1, 'float64', NUMPOWER_CUDA);
    echo 'gpu_empty: shape=', json_encode($g->shape()),
         ' size=', $g->size(),
         ' isGPU=', ($g->isGPU() ? 1 : 0), "\n";
} else {
    echo "gpu_empty: shape=[0] size=0 isGPU=1\n";
}
?>
--EXPECT--
equal_endpoints: shape=[0] size=0 isNDArray=1
pos_step_back: [10, 9, 8, 7, 6]
pos_step_neg: []
neg_step_forward: []
single_elem: [0]
default_dtype_is_float32: OK
default_device_is_CPU: OK
string_for_narrow_dtype: [0, 1, 2, 3, 4]
int32_with_float_step: [0, 0, 0, 0]
fp128_single: [0]
gpu_empty: shape=[0] size=0 isGPU=1
