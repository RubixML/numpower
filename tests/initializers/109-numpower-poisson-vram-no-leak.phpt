--TEST--
NumPower::poisson() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress: many poisson() allocations across every dtype, with int /
   float / string lam forms. Each iteration's NDArray is overwritten —
   every VRAM slot must be released. Any imbalance surfaces as
   `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN.

   Every non-uint32 dtype takes a transient u32 scratch via `vmalloc`
   plus, for fp4 / fp8, a second f32 scratch. Both scratches must be
   freed before each call returns. */

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
    if ($dt === 'float4')      { $lam = 1.0; }
    elseif ($dt === 'float8')  { $lam = 2.0; }
    elseif ($dt === 'int8')    { $lam = 30.0; }
    elseif ($dt === 'uint8')   { $lam = 100.0; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $lam = 1000.0; }
    elseif ($dt === 'uint64')  { $lam = '100000'; }
    elseif ($dt === 'float128'){ $lam = '50.0'; }
    else                       { $lam = 5.0; }

    foreach ($cases as $shape) {
        for ($i = 0; $i < 3; $i++) {
            $a = NumPower::poisson($shape, $lam, $dt, NUMPOWER_CUDA);
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
