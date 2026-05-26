--TEST--
NumPower::standardNormal() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress: many standardNormal() allocations across every dtype, both
   odd and even sample counts. Each iteration's NDArray is overwritten —
   every VRAM slot must be released. Any imbalance surfaces as
   `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN.

   The fp128 / int64 / uint64 paths each take their own VRAM scratch
   inside NDArray_Normal (cuRAND scratch + cuda_cast for the small
   dtypes, on-host stage for uint64). Every scratch must be freed
   before the function returns. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$cases = [
    [8],       /* small, even */
    [7],       /* small, odd  */
    [0],       /* empty       */
    [],        /* 0-D scalar  */
    [128],     /* small, even */
    [3, 5],    /* multi-dim, odd total */
    [4, 8],    /* multi-dim, even total */
];

foreach ($dtypes as $dt) {
    foreach ($cases as $shape) {
        for ($i = 0; $i < 3; $i++) {
            $a = NumPower::standardNormal($shape, $dt, NUMPOWER_CUDA);
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
