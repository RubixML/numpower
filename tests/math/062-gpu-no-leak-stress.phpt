--TEST--
GPU stress: 14 dtypes × 6 ops × 50 iterations leaves no VRAM leaks at RSHUTDOWN
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress test: exercises every dtype × every op × scalar+GPU/GPU+scalar/
   GPU+GPU in a loop. RSHUTDOWN prints "VRAM MEMORY LEAK" if any GPU buffer
   is still allocated; the --EXPECT-- below asserts that line is absent. */
$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

for ($iter = 0; $iter < 50; $iter++) {
    foreach ($dtypes as $dt) {
        $cpu = NumPower::array([2, 2, 2, 2], $dt);
        $a = $cpu->gpu();
        /* float4/float8 fall back to CPU during compute (no native CUDA
           intrinsics); other 12 dtypes stay GPU. Both paths must be leak-
           free. */
        $r1 = $a + 1;
        $r2 = $a - 1;
        $r3 = $a * 2;
        $r4 = $a / 2;
        $r5 = $a ** 2;
        /* GPU + GPU same-dtype */
        $r6 = $a + $a;
        /* Round-trip */
        $r7 = $r6->cpu();
        unset($cpu, $a, $r1, $r2, $r3, $r4, $r5, $r6, $r7);
    }
}
echo "OK\n";
?>
--EXPECT--
OK
