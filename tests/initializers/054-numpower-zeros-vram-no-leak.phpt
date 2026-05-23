--TEST--
NumPower::zeros() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress the GPU init path: hundreds of zeros() calls across every dtype +
   a mix of shapes (incl. zero-element). Each iteration's NDArray is
   reassigned, so the previous allocation must release its VRAM slot. The
   NDARRAY_VCHECK runtime walks `MAIN_MEM_STACK.totalGPUAllocated` at
   RSHUTDOWN; any positive count prints `VRAM MEMORY LEAK: leaked N
   array(s)` and breaks the EXPECT. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$shapes = [[1], [16], [3, 5], [2, 3, 4], [0], [5, 0]];

foreach ($dtypes as $dt) {
    foreach ($shapes as $shape) {
        for ($i = 0; $i < 5; $i++) {
            $a = NumPower::zeros($shape, $dt, NUMPOWER_CUDA);
            /* Force a single-element D2H read for non-empty shapes so any
               half-initialised buffer would surface as a crash. */
            if ($a->size() > 0) {
                $cpu = $a->cpu();
                $cpu = null;
            }
            $a = null;
        }
    }
}

echo "done\n";
?>
--EXPECT--
done
