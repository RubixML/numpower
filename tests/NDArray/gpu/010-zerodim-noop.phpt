--TEST--
NDArray::gpu()/cpu() on a 0-dim array still returns a scalar correctly
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* 0-dim NDArrays are a corner case: the legacy API collapses them into a
   PHP scalar at the gpu()/cpu() boundary. This test guards that contract
   so the same-device no-op refactor doesn't regress it. */

/* CPU 0-dim → cpu(): scalar comes back, no exception. */
$a   = new NDArray(3.14, 'float64');
$out = $a->cpu();
$ok1 = is_float($out) && abs($out - 3.14) < 1e-9;

/* CPU 0-dim → gpu(): real CPU→GPU transfer; the value materialises back
   into a PHP scalar via ndarray_init_new_object. */
$b   = new NDArray(2.5, 'float32');
$g   = $b->gpu();
$ok2 = is_float($g) && abs($g - 2.5) < 1e-6;

/* Float32 path. */
$c   = (new NDArray(7.0, 'float32'))->gpu();
$ok3 = is_float($c) && abs($c - 7.0) < 1e-6;

echo ($ok1 && $ok2 && $ok3) ? "OK\n" : "FAIL\n";
?>
--EXPECT--
OK
