--TEST--
NumPower::full() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress test: hundreds of full() calls across every dtype + a mix of
   shapes (incl. zero-element + 0-D). Each iteration's NDArray is
   overwritten; the previous slot must release its VRAM. Any imbalance
   surfaces as `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN.

   The wide-dtype string fills (fp128 / int64 / uint64) cover the
   stringification → strtoll / strtoull / strtoflt128 path inside the
   encoder so a leak there is also caught. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$shapes = [[], [1], [16], [3, 5], [2, 3, 4], [0], [5, 0]];

foreach ($dtypes as $dt) {
    $string_io = in_array($dt, ['float128', 'int64', 'uint64'], true);
    $value = $string_io ? '1' : 1;
    foreach ($shapes as $shape) {
        for ($i = 0; $i < 5; $i++) {
            $a = NumPower::full($shape, $value, $dt, NUMPOWER_CUDA);
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
