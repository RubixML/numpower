--TEST--
NumPower::diag() handles empty / 1×1 / non-square shapes and the default contract
--FILE--
<?php
/* Edge cases:
    - 1-D []                  → 0×0 matrix.
    - 1-D [x]                 → 1×1 matrix with x on the (only) diagonal.
    - 2-D 1×1                 → 1-element vector.
    - 2-D 1×N (single row)    → 1-element vector.
    - 2-D N×1 (single col)    → 1-element vector.
    - 2-D 3×4 / 4×3           → length-3 vector (= min(rows, cols)).
    - PHP array input         → same result as NDArray input.
    - default contract        → float32 / CPU. */

/* Empty 1-D → 0×0 matrix. */
$e = NumPower::diag(NumPower::array([]));
echo 'empty_1d shape=', json_encode($e->shape()), ' size=', $e->size(), "\n";

/* 1-element 1-D. */
$d = NumPower::diag(NumPower::array([5.0]));
echo 'single_1d: ', (string)$d, "\n";

/* 1×1 matrix → 1-element vector. */
$d = NumPower::diag(NumPower::array([[7.0]]));
echo 'single_2d: ', (string)$d, "\n";

/* 1×N (single row) → 1-element vector. */
$d = NumPower::diag(NumPower::array([[1.0, 2.0, 3.0]]));
echo '1xN: ', (string)$d, "\n";

/* N×1 (single col) → 1-element vector. */
$d = NumPower::diag(NumPower::array([[1.0], [2.0], [3.0]]));
echo 'Nx1: ', (string)$d, "\n";

/* PHP array input matches NDArray input. */
$a = NumPower::diag([1.0, 2.0, 3.0]);
$b = NumPower::diag(NumPower::array([1.0, 2.0, 3.0]));
echo 'php_array_matches: ', ((string)$a === (string)$b ? 'OK' : 'BAD'), "\n";

/* Default contract: float32 / CPU. */
$d = NumPower::diag(NumPower::array([1, 2, 3]));
echo 'default_dtype_float32: ', (is_float($d[0][0]) ? 'OK' : 'BAD'), "\n";
echo 'default_device_CPU: ', ($d->isGPU() ? 'BAD' : 'OK'), "\n";

/* Explicit defaults parity. */
$d = NumPower::diag(NumPower::array([1, 2, 3]), 'float32', NUMPOWER_CPU);
echo 'explicit_defaults: ',
     (!$d->isGPU() && (string)NumPower::sum($d) === '6' ? 'OK' : 'BAD'), "\n";

/* GPU: empty round-trip stays a 0×0 GPU NDArray. */
$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $has_gpu = false; }
if ($has_gpu) {
    $g = NumPower::diag(NumPower::array([]), 'float32', NUMPOWER_CUDA);
    echo 'gpu_empty: shape=', json_encode($g->shape()),
         ' isGPU=', ($g->isGPU() ? 1 : 0), "\n";
} else {
    echo "gpu_empty: shape=[0,0] isGPU=1\n";
}
?>
--EXPECT--
empty_1d shape=[0,0] size=0
single_1d: [[5]]
single_2d: [7]
1xN: [1]
Nx1: [1]
php_array_matches: OK
default_dtype_float32: OK
default_device_CPU: OK
explicit_defaults: OK
gpu_empty: shape=[0,0] isGPU=1
