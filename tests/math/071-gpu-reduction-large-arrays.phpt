--TEST--
GPU reductions handle multi-block grids correctly via atomic accumulation
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The atomic-only kernels run with blockSize=256 so any array larger
   than 256 elements forces multi-block execution. Per-block contention
   on the atomic accumulator must not affect correctness — verify by
   feeding arrays of growing size and comparing to the analytic answer. */
foreach ([300, 1024, 5000, 100000] as $n) {
    $vals = range(1, $n);  /* sum = n*(n+1)/2 */
    $expected_sum = $n * ($n + 1) / 2;
    foreach (['float32', 'float64', 'int32', 'int64'] as $t) {
        $a = (new NDArray($vals, $t))->gpu();
        $got = (float) NumPower::sum($a);
        /* float32 carries ~7 decimal digits; relative tolerance. */
        $rel = abs($got - $expected_sum) / abs($expected_sum);
        $eps = ($t === 'float32') ? 1e-5 : 1e-12;
        if ($rel > $eps) {
            echo "n=$n $t sum: FAIL want=$expected_sum got=$got rel=$rel\n";
            exit;
        }
        $mx = (float) NumPower::max($a);
        $mn = (float) NumPower::min($a);
        if ($mx != $n || $mn != 1) {
            echo "n=$n $t min/max: FAIL min=$mn max=$mx\n";
            exit;
        }
    }
}
echo "ok\n";
?>
--EXPECT--
ok
