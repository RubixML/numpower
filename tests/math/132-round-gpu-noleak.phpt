--TEST--
GPU NumPower::round (every dtype × several precisions × many iterations) leaves no VRAM leak at RSHUTDOWN
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
/* Stress the GPU round path's memory bookkeeping: each call clones the
   input via NDArray_Copy / NDArray_AsType (fp4/fp8) and rounds in place,
   so every allocation must be paired with an NDArray_FREE and the kernel
   must not leave scratch behind. Run many iterations across every dtype
   and precision; a single leaked device buffer makes the RSHUTDOWN
   `vmemcheck` print `VRAM MEMORY LEAK: leaked N array(s)`, which would
   appear before DONE and fail the --EXPECT-- match. */

$float_dtypes = ['float4','float8','float16','float32','float64','float128'];
$int_dtypes   = ['int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

for ($iter = 0; $iter < 25; $iter++) {
    foreach ($float_dtypes as $dt) {
        $a = NumPower::array([0.5, 1.5, 2.5, 3.49, -2.5], $dt)->gpu();
        foreach ([0, 2, -1] as $dec) {
            $r = NumPower::round($a, $dec);
            unset($r);
        }
        unset($a);
    }
    foreach ($int_dtypes as $dt) {
        $a = NumPower::array([12, 25, 37], $dt)->gpu();
        $r = NumPower::round($a, -1);   /* integer identity copy on GPU */
        unset($r);
        unset($a);
    }
    /* multi-dim + empty + 0-D on GPU */
    $m = NumPower::array([[0.5, 1.5], [2.5, 3.5]], 'float64')->gpu();
    unset($m);
    $r = NumPower::round(NumPower::zeros([0, 4], 'float32')->gpu(), 2);
    unset($r);
    $s = NumPower::round(NumPower::array(2.5, 'float64')->gpu(), 0);
    unset($s);
}

echo "DONE\n";
?>
--EXPECT--
DONE
