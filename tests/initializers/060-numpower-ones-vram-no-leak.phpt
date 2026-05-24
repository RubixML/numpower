--TEST--
NumPower::ones() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress the GPU ones() path through the same iteration pattern that
   triggered the original RSHUTDOWN VRAM leak: hundreds of allocations
   across every dtype + a mix of shapes (incl. zero-element). Each
   iteration's NDArray is overwritten, so the previous slot must release
   its VRAM. NDARRAY_VCHECK reports any imbalance via vmemcheck(). */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$shapes = [[1], [16], [3, 5], [2, 3, 4], [0], [5, 0]];

foreach ($dtypes as $dt) {
    foreach ($shapes as $shape) {
        for ($i = 0; $i < 5; $i++) {
            $a = NumPower::ones($shape, $dt, NUMPOWER_CUDA);
            /* Force a single-element D2H read for non-empty shapes so any
               half-initialised buffer surfaces as a crash or wrong value. */
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
