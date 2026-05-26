--TEST--
NumPower::randomBinomial() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress: many randomBinomial() allocations across every dtype, with
   varied (n, p). Each iteration's NDArray is overwritten — every VRAM
   slot must be released. Any imbalance surfaces as `VRAM MEMORY LEAK`
   at RSHUTDOWN.

   Non-uint32 dtypes take a transient u32 scratch via `vmalloc`; fp4
   and fp8 add a second f32 scratch. Both scratches must be freed
   before each call returns. The n=0 / p=0 short-circuit and the
   p=1.0 path are also exercised.

   The legacy CPU implementation had a `(float)rand() / (float)RAND_MAX`
   that could yield exactly 1.0; the new GPU kernel uses
   `1 - curand_uniform` so the comparison `u < p` honours the closed-
   open convention, and `p == 1.0` always succeeds. */

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

$scenarios = [
    [10, 0.5],   /* typical    */
    [50, 0.3],   /* asymmetric */
    [0, 0.5],    /* degenerate n=0 (cudaMemset short-circuit) */
    [10, 0.0],   /* degenerate p=0 (cudaMemset short-circuit) */
    [10, 1.0],   /* p=1.0 (kernel runs but every trial succeeds) */
];

foreach ($dtypes as $dt) {
    foreach ($cases as $shape) {
        foreach ($scenarios as [$n, $p]) {
            for ($i = 0; $i < 2; $i++) {
                $a = NumPower::randomBinomial($shape, $n, $p, $dt, NUMPOWER_CUDA);
                if ($a->size() > 0) {
                    $cpu = $a->cpu();
                    $cpu = null;
                }
                $a = null;
            }
        }
    }
}

echo "done\n";
?>
--EXPECT--
done
