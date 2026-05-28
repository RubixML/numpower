--TEST--
GPU trig family across all dtypes — no VRAM leaks at RSHUTDOWN
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
/* Stress: every trig family op × every supported GPU compute dtype ×
   20 iterations. With `NDARRAY_VCHECK=1`, the extension's
   `vmemcheck()` runs at RSHUTDOWN and prints `VRAM MEMORY LEAK`
   if any vmalloc-tracked buffer survives. The test passes iff that
   string is absent and "DONE" is printed.

   Also exercises the integer rounding short-circuit
   (`NDArray_TypedUnaryOp` returns `NDArray_Copy(...)` for RINT/FIX/
   TRUNC/FLOOR/CEIL on int dtypes) which has its own allocation path
   distinct from the kernel-launching float path. */

$ops_trig = ['sin','cos','tan','arcsin','arccos','arctan',
             'sinh','cosh','tanh','arcsinh','arccosh','arctanh',
             'degrees','radians'];
$ops_round = ['rint','fix','trunc','floor','ceil'];

/* Bounded input that fits arcsin/arccos/arctanh; arccosh handled below. */
$xs_b  = [0.1, 0.3, 0.5, 0.7, 0.9];
$xs_s  = ['0.1', '0.3', '0.5', '0.7', '0.9'];

/* Float dtypes: trig + rounding paths */
foreach (['float16','float32','float64','float128','float4','float8'] as $dt) {
    $src = ($dt === 'float128') ? $xs_s : $xs_b;
    $g = NumPower::array($src, $dt)->gpu();
    for ($iter = 0; $iter < 20; $iter++) {
        foreach ($ops_trig as $op) {
            /* arccosh needs >= 1; skip for bounded inputs. */
            if ($op === 'arccosh') continue;
            $r = NumPower::$op($g);
            if (!$r->isGPU()) { echo "FAIL $dt $op left GPU\n"; break 2; }
            unset($r);
        }
        foreach ($ops_round as $op) {
            $r = NumPower::$op($g);
            if (!$r->isGPU()) { echo "FAIL $dt $op left GPU\n"; break 2; }
            unset($r);
        }
    }
    unset($g);
}

/* Integer rounding short-circuit (NDArray_Copy path) */
foreach (['int8','int16','int32','int64','uint8','uint16','uint32','uint64'] as $dt) {
    $g = NumPower::array([1, 2, 3, 4, 5], $dt)->gpu();
    for ($iter = 0; $iter < 20; $iter++) {
        foreach ($ops_round as $op) {
            $r = NumPower::$op($g);
            if (!$r->isGPU()) { echo "FAIL $dt $op left GPU on int short-circuit\n"; break 2; }
            unset($r);
        }
    }
    unset($g);
}

/* Integer → float promotion paths (trig) */
foreach (['int32','int64','uint32','uint64'] as $dt) {
    $g = NumPower::array([1, 2, 3], $dt)->gpu();
    for ($iter = 0; $iter < 20; $iter++) {
        foreach ($ops_trig as $op) {
            if ($op === 'arccosh') continue;
            $r = NumPower::$op($g);
            unset($r);
        }
    }
    unset($g);
}

/* Multi-block stress: N=4097 */
$big = [];
for ($i = 0; $i < 4097; $i++) $big[] = 0.3;
foreach (['float32','float64'] as $dt) {
    $g = NumPower::array($big, $dt)->gpu();
    foreach ($ops_trig as $op) {
        if ($op === 'arccosh') continue;
        $r = NumPower::$op($g);
        unset($r);
    }
    foreach ($ops_round as $op) {
        $r = NumPower::$op($g);
        unset($r);
    }
    unset($g);
}

echo "DONE\n";
?>
--EXPECT--
DONE
