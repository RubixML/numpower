--TEST--
NumPower::arctan2 on GPU across all dtypes — no VRAM leaks at RSHUTDOWN
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
/* Stress arctan2 on the GPU across every dtype × broadcasting modes ×
   iterations. With NDARRAY_VCHECK=1 the extension's vmemcheck() runs at
   RSHUTDOWN and prints `VRAM MEMORY LEAK` if any vmalloc-tracked buffer
   survives. The test passes iff that string is absent and "DONE" prints.

   arctan2 allocates several transient GPU buffers per call (the AsType casts
   to the common compute dtype, the broadcast offset buffer via vmalloc/vfree,
   the result, and the cast-back for fp16 / fp4 / fp8), so a missing vfree on
   any of them would surface here. */

$pairs_signed   = [[1.0, -1.0, 0.5, -0.5], [1.0, 1.0, -0.5, 0.5]];
$pairs_unsigned = [[0, 1, 2, 3], [1, 0, 4, 2]];

$float_dts = ['float16', 'float32', 'float64', 'float128', 'float4', 'float8'];
$int_dts   = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'];

foreach ($float_dts as $dt) {
    $src = ($dt === 'float128')
        ? [['1.0', '-1.0', '0.5', '-0.5'], ['1.0', '1.0', '-0.5', '0.5']]
        : $pairs_signed;
    $x = NumPower::array($src[0], $dt)->gpu();
    $y = NumPower::array($src[1], $dt)->gpu();
    for ($i = 0; $i < 25; $i++) {
        $r = NumPower::arctan2($x, $y);
        if (!$r->isGPU()) { echo "FAIL $dt left GPU\n"; break 2; }
        unset($r);
        /* scalar-broadcast path (allocates a broadcast buffer) */
        $rb = NumPower::arctan2($x, 1.0);
        unset($rb);
    }
    unset($x, $y);
}

foreach ($int_dts as $dt) {
    $x = NumPower::array($pairs_unsigned[0], $dt)->gpu();
    $y = NumPower::array($pairs_unsigned[1], $dt)->gpu();
    for ($i = 0; $i < 25; $i++) {
        $r = NumPower::arctan2($x, $y);
        if (!$r->isGPU()) { echo "FAIL $dt left GPU\n"; break 2; }
        unset($r);
    }
    unset($x, $y);
}

/* Broadcast + multi-block stress on the largest float path */
$N = 4097; $big = [];
for ($i = 0; $i < $N; $i++) $big[$i] = (($i % 9) - 4) * 0.25;
foreach (['float32', 'float64'] as $dt) {
    $g = NumPower::array($big, $dt)->gpu();
    $col = NumPower::array([0.5], $dt)->gpu();   /* 1-elem broadcast denominator */
    for ($i = 0; $i < 10; $i++) {
        $r = NumPower::arctan2($g, $col);
        unset($r);
        $r2 = NumPower::arctan2($g, 2.0);
        unset($r2);
    }
    unset($g, $col);
}

echo "DONE\n";
?>
--EXPECT--
DONE
