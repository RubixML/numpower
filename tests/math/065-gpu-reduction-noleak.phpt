--TEST--
GPU reductions (sum/prod/min/max/mean) do not leak their scratch buffer
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* `cuda_sum_float`, `cuda_prod_float`, `cuda_sum_double`,
   `cuda_max_float`, and `cuda_min_float` each allocated a scratch
   buffer with `cudaMalloc` and never freed it. They now use
   `vmalloc`/`vfree`, so the VCHECK hook at RSHUTDOWN flags any leak
   via "VRAM MEMORY LEAK" — the test passes only if it stays silent.

   We exercise every reducer on both float32 and float64 in a loop so a
   single missing `vfree` accumulates enough imbalance to be noticed. */
$dtypes = ['float32', 'float64'];
foreach ($dtypes as $t) {
    for ($i = 0; $i < 100; $i++) {
        $a = (new NDArray([1.0, 2.0, 3.0, 4.0, 5.0], $t))->gpu();
        $s = NumPower::sum($a);
        $p = NumPower::prod($a);
        $mx = NumPower::max($a);
        $mn = NumPower::min($a);
        $av = NumPower::mean($a);
        unset($a, $s, $p, $mx, $mn, $av);
    }
}
echo "ok\n";
?>
--EXPECT--
ok
