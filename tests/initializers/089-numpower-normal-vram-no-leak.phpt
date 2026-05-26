--TEST--
NumPower::normal() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress: many normal() allocations across every dtype, both odd and
   even sample counts, with int/float/string loc/scale forms (so the
   string parsers also flow). Each iteration's NDArray is overwritten —
   every VRAM slot must be released. Any imbalance surfaces as
   `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN.

   The fp128 / int64 / uint64 string paths also exercise the
   coerce_zval_to_<wide> helpers and the on-GPU DD-affine kernel for
   fp128. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$cases = [
    [8],       /* small, even */
    [7],       /* small, odd  */
    [0],       /* empty       */
    [128],     /* small, even */
    [3, 5],    /* multi-dim, odd total */
    [4, 8],    /* multi-dim, even total */
];

foreach ($dtypes as $dt) {
    $string_io = in_array($dt, ['float128', 'uint64'], true);
    if ($dt === 'float4') { $loc = 1.0; $scale = 0.5; }
    elseif ($dt === 'float8') { $loc = 2.0; $scale = 1.0; }
    elseif ($dt === 'int8') { $loc = 0; $scale = 20; }
    elseif ($dt === 'uint8') { $loc = 128; $scale = 30; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $loc = 1000; $scale = 100; }
    elseif ($dt === 'uint64') { $loc = '1000000'; $scale = '100'; }
    elseif ($dt === 'float128') { $loc = '0.0'; $scale = '1.0'; }
    else { $loc = 0.0; $scale = 1.0; }

    foreach ($cases as $shape) {
        for ($i = 0; $i < 3; $i++) {
            $a = NumPower::normal($shape, $loc, $scale, $dt, NUMPOWER_CUDA);
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
