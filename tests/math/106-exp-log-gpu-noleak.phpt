--TEST--
GPU exp/log family across all dtypes — no VRAM leaks at RSHUTDOWN
--SKIPIF--
<?php
try {
    $a = NumPower::array([1.0])->gpu();
    if (!$a->isGPU()) die("skip GPU not available");
} catch (Throwable $t) {
    die("skip GPU not available: " . $t->getMessage());
}
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress test: every transcendental op × every supported GPU compute
   dtype × 25 iterations. With NDARRAY_VCHECK=1, vmemcheck() prints
   "VRAM MEMORY LEAK" at RSHUTDOWN if any vmalloc'd buffer survives.
   The expected output therefore ends with `vmem OK` (the wrapper line)
   and absolutely no "VRAM MEMORY LEAK". */

$ops = ['exp','exp2','expm1','log','log1p','log2','log10','logb'];

/* Compute dtypes the GPU dispatcher routes natively. fp4/fp8 stay on
   GPU but cast through fp32 — also exercised below. Integer dtypes
   promote to fp32/fp64 — covered too. */
$dtypes = [
    'float16','float32','float64','float128',
    'float4','float8',
    'int8','int16','int32','int64',
    'uint8','uint16','uint32','uint64',
];

$xs_num   = [1.0, 2.0, 4.0, 8.0, 16.0];
$xs_str   = ['1.0', '2.0', '4.0', '8.0', '16.0'];

foreach ($dtypes as $dt) {
    /* fp128 needs string input for precision; everything else accepts numerics. */
    $src = ($dt === 'float128') ? $xs_str : $xs_num;
    $arr = NumPower::array($src, $dt)->gpu();

    for ($iter = 0; $iter < 25; $iter++) {
        foreach ($ops as $op) {
            $r = NumPower::$op($arr);
            if (!$r->isGPU()) {
                echo "FAIL $dt $op iter=$iter left GPU\n";
                break 2;
            }
            unset($r);   /* triggers buffer dec_ref */
        }
    }
}

echo "DONE\n";
?>
--EXPECT--
DONE
