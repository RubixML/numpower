--TEST--
NumPower::uniform() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress: many uniform() allocations across every dtype, with int /
   float / string low/high forms (so the string parsers also flow).
   Each iteration's NDArray is overwritten — every VRAM slot must be
   released. Any imbalance surfaces as `VRAM MEMORY LEAK: leaked N
   array(s)` at RSHUTDOWN.

   The fp128 / int64 / uint64 paths each take their own VRAM scratch
   inside NDArray_Uniform (cuRAND scratch + cuda_cast for the small
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
    if ($dt === 'float4') { $low = 0.0; $high = 1.0; }
    elseif ($dt === 'float8') { $low = 0.0; $high = 2.0; }
    elseif ($dt === 'int8') { $low = 0; $high = 100; }
    elseif ($dt === 'uint8') { $low = 0; $high = 255; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $low = 0; $high = 1000; }
    elseif ($dt === 'uint64') { $low = '1000000'; $high = '2000000'; }
    elseif ($dt === 'float128') { $low = '0.0'; $high = '1.0'; }
    else { $low = 0.0; $high = 1.0; }

    foreach ($cases as $shape) {
        for ($i = 0; $i < 3; $i++) {
            $a = NumPower::uniform($shape, $low, $high, $dt, NUMPOWER_CUDA);
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
