--TEST--
GPU reductions still produce correct values after the leak fix
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The leak fix swapped raw cudaMalloc/cudaFree for vmalloc/vfree on
   the reduction scratch buffer. Since vmalloc and cudaMalloc both
   return a regular device pointer, the kernel must see no behavior
   change. Verifies on a boundary-rich input (negative, zero, fractional)
   that sum/max/min/mean still match the analytic answer on float32.

   prod on a zero-containing input is not verified here because the
   underlying cuda_prod_float kernel has a pre-existing bug that returns
   the seed value (1) rather than the actual product when 0.0 is present
   — unrelated to this leak fix. Non-zero-only inputs would test prod
   but obscure the boundary-value coverage we care about. float64 isn't
   tested either: NumPower::sum / prod / max / min route every dtype
   through the float32 helpers (NDArray_Sum_Float, NDArray_Float_Prod),
   so float64 inputs produce dtype-mismatched results regardless of the
   leak fix. */
$vals = [-2.5, -1.0, 0.0, 1.0, 2.5, 3.5, 4.0, 5.5];
$expect = [
    'sum'  => array_sum($vals),
    'max'  => max($vals),
    'min'  => min($vals),
    'mean' => array_sum($vals) / count($vals),
];

$a = (new NDArray($vals, 'float32'))->gpu();
$got = [
    'sum'  => NumPower::sum($a),
    'max'  => NumPower::max($a),
    'min'  => NumPower::min($a),
    'mean' => NumPower::mean($a),
];
$eps = 1e-4;
foreach ($expect as $op => $want) {
    $g = (float)$got[$op];
    $diff = abs($g - $want);
    $rel = $want != 0.0 ? $diff / abs($want) : $diff;
    if ($rel > $eps) {
        echo "$op: FAIL want=$want got=$g rel=$rel\n";
        exit;
    }
}
echo "ok\n";
?>
--EXPECT--
ok
