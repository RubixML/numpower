--TEST--
GPU reductions on 2-D / 3-D arrays flatten correctly and stay on GPU
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The reductions accept any ndim — they read the buffer as a flat
   sequence of numElements. Multi-dim arrays should give the same result
   as the equivalent flattened 1-D array. Verifies the dispatch / kernel
   doesn't assume ndim == 1 anywhere. */

/* 2-D float32 (3x4 grid). */
$a2 = (new NDArray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 'float32'))->gpu();
$want_sum  = (1+2+3+4 + 5+6+7+8 + 9+10+11+12);   /* 78 */
$want_max  = 12;
$want_min  = 1;
$want_mean = $want_sum / 12.0;
$ok = NumPower::sum($a2) == $want_sum
   && NumPower::max($a2) == $want_max
   && NumPower::min($a2) == $want_min
   && abs(NumPower::mean($a2) - $want_mean) < 1e-9;
echo "2-D float32: ", $ok ? "ok\n" : "FAIL sum=" . NumPower::sum($a2) . "\n";

/* 3-D int32 (2x2x3). */
$vals = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]],
];
$a3 = (new NDArray($vals, 'int32'))->gpu();
$ok = NumPower::sum($a3) == 78
   && NumPower::max($a3) == 12
   && NumPower::min($a3) == 1
   && abs(NumPower::mean($a3) - 6.5) < 1e-9;
echo "3-D int32: ", $ok ? "ok\n" : "FAIL sum=" . NumPower::sum($a3) . "\n";

/* 2-D float64. */
$a2d = (new NDArray([[0.5, 1.5], [2.5, 3.5]], 'float64'))->gpu();
$ok = abs(NumPower::sum($a2d) - 8.0) < 1e-12
   && abs(NumPower::prod($a2d) - 6.5625) < 1e-12;
echo "2-D float64: ", $ok ? "ok\n" : "FAIL sum=" . NumPower::sum($a2d) . " prod=" . NumPower::prod($a2d) . "\n";
?>
--EXPECT--
2-D float32: ok
3-D int32: ok
2-D float64: ok
