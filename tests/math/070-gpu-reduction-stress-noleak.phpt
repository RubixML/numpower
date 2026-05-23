--TEST--
GPU reductions across every dtype do not leak VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Every reduction in NDArray_Reduce_* allocates a one-double GPU
   accumulator via vmalloc and frees it via vfree before returning.
   Mismatched calls would increment MAIN_MEM_STACK.totalGPUAllocated;
   the vmemcheck() hook fired by RSHUTDOWN when NDARRAY_VCHECK=1 prints
   "VRAM MEMORY LEAK: leaked N array(s)" to stderr, which then fails
   the EXPECT block. The test passes only if VCHECK stays silent across
   every dtype × every reducer × 30 iterations.

   Note: gc_collect_cycles / memory_get_usage are deliberately NOT used
   here — they track PHP's Zend MM heap, not VRAM. The VCHECK hook is
   the only mechanism that actually counts unfreed device buffers. */
$dtypes = [
    'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
];
foreach ($dtypes as $t) {
    for ($i = 0; $i < 30; $i++) {
        $a = (new NDArray([1, 2, 3, 4, 5], $t))->gpu();
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
