--TEST--
NumPower hyperbolic family on GPU across all dtypes — no VRAM leaks at RSHUTDOWN
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
/* Stress the six hyperbolic ops on the GPU across every dtype × iterations.
   With NDARRAY_VCHECK=1 the extension's vmemcheck() runs at RSHUTDOWN and
   prints `VRAM MEMORY LEAK` if any vmalloc-tracked buffer survives. The test
   passes iff that string is absent and "DONE" prints.

   Each unary call clones the input to a work buffer (NDArray_Copy, on-device)
   and runs the kernel in place — a missing free on the work buffer (or, for
   fp4/fp8 inputs, the float32 round-trip cast) would surface here. The
   bare-string intake path additionally allocates an owned 0-D scalar that
   must be freed by ndarray_release_unary_input. */

$OPS = ['sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh'];

/* Per-op in-domain seed values (arccosh ≥ 1, arctanh in (−1, 1)). */
$seed = [
    'sinh' => 0.5, 'cosh' => 0.5, 'tanh' => 0.5,
    'arcsinh' => 0.5, 'arccosh' => 1.5, 'arctanh' => 0.5,
];
$float_dts = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128'];
$int_dts   = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($OPS as $op) {
    $v = $seed[$op];
    foreach ($float_dts as $dt) {
        $src = ($dt === 'float128')
            ? new NDArray([(string)$v, (string)$v, (string)$v, (string)$v], $dt)
            : new NDArray([$v, $v, $v, $v], $dt);
        $g = $src->gpu();
        for ($i = 0; $i < 20; $i++) {
            $r = NumPower::$op($g);
            if (!$r->isGPU()) { echo "FAIL $op $dt left GPU\n"; break 3; }
            unset($r);
        }
        unset($src, $g);
    }
    /* integer dtypes: arccosh needs ≥1, others fine at the integer seed. */
    $iv = (int)ceil($v);
    if ($op === 'arctanh') $iv = 0;          /* only integral in-domain value */
    foreach ($int_dts as $dt) {
        $g = (new NDArray([$iv, $iv, $iv, $iv], $dt))->gpu();
        for ($i = 0; $i < 20; $i++) {
            $r = NumPower::$op($g);
            if (!$r->isGPU()) { echo "FAIL $op $dt left GPU\n"; break 3; }
            unset($r);
        }
        unset($g);
    }
}

/* Multi-block stress (N > one CUDA block) on the heaviest fp128 DD path. */
$N = 4097; $big = [];
for ($i = 0; $i < $N; $i++) $big[$i] = (string)((($i % 100) / 100.0) * 0.9 + 0.05); /* (0,1) */
$g = (new NDArray($big, 'float128'))->gpu();
for ($i = 0; $i < 5; $i++) {
    foreach (['sinh', 'cosh', 'tanh', 'arcsinh', 'arctanh'] as $op) {
        $r = NumPower::$op($g);
        unset($r);
    }
}
unset($g);

/* Bare numeric-string intake (owned 0-D scalar release path). */
for ($i = 0; $i < 50; $i++) {
    $r = NumPower::tanh('0.5');               /* decimal → fp128 scalar */
    unset($r);
}

echo "DONE\n";
?>
--EXPECT--
DONE
