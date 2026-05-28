--TEST--
sign(NaN) and clip NaN-bounds / lo > hi semantics (GPU)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU half of 101-sign-clip.phpt. Skipped without a CUDA device —
   same semantic contract, same fp128 DD emulation path. */

function approx($a, $b, $tol = 1e-5) {
    if (is_nan((float)$a) && is_nan((float)$b)) return true;
    return abs((float)$a - (float)$b) <= $tol;
}

foreach (['float32', 'float64', 'float16'] as $dt) {
    $g     = NumPower::array([NAN, INF, -INF, 0.0, -3.0, 2.0], $dt)->gpu();
    $sign  = NumPower::sign($g);
    if (!$sign->isGPU()) { echo "FAIL $dt sign left GPU\n"; continue; }
    $s_cpu = $sign->cpu()->toArray();
    echo "GPU/$dt sign(NaN) is_nan: ", is_nan($s_cpu[0]) ? "yes" : "no", "\n";
    echo "GPU/$dt sign(+Inf): ", $s_cpu[1], "\n";
    echo "GPU/$dt sign(-Inf): ", $s_cpu[2], "\n";
}

$g128 = NumPower::array(['nan', '0', '-1', '1e30'], 'float128')->gpu();
$sg   = NumPower::sign($g128);
if (!$sg->isGPU()) { echo "FAIL fp128 sign left GPU\n"; }
else {
    $a = $sg->cpu()->toArray();
    echo "GPU/fp128 sign(nan): ", $a[0], "\n";
    echo "GPU/fp128 sign(0): ",   $a[1], "\n";
    echo "GPU/fp128 sign(-1): ",  $a[2], "\n";
}

foreach (['float32', 'float64', 'float16'] as $dt) {
    $g = NumPower::array([NAN, 3.0], $dt)->gpu();
    $r = NumPower::clip($g, 0.0, 5.0);
    $a = $r->cpu()->toArray();
    echo "GPU/$dt clip(NaN, 0, 5) is_nan: ", is_nan($a[0]) ? "yes" : "no",
         "; clip(3, 0, 5)=", $a[1], "\n";

    $g2 = NumPower::array([-1.0, 7.0, 100.0], $dt)->gpu();
    $r2 = NumPower::clip($g2, 10.0, 5.0);
    $a2 = $r2->cpu()->toArray();
    $ok = approx($a2[0], 5.0, 1e-3) && approx($a2[1], 5.0, 1e-3)
       && approx($a2[2], 5.0, 1e-3);
    echo "GPU/$dt clip(any, lo>hi)=hi: ", $ok ? "OK" : "FAIL " . json_encode($a2), "\n";
}

$gf = NumPower::array(['nan', '3', '100'], 'float128')->gpu();
$rf = NumPower::clip($gf, '0', '5')->cpu()->toArray();
echo "GPU/fp128 clip(nan,0,5): ", $rf[0], "\n";
echo "GPU/fp128 clip(3,0,5): ", $rf[1], "\n";
echo "GPU/fp128 clip(100,0,5): ", $rf[2], "\n";

$rf2 = NumPower::clip(NumPower::array(['3', '7'], 'float128')->gpu(),
                      '10', '5')->cpu()->toArray();
echo "GPU/fp128 clip(3,lo=10,hi=5): ", $rf2[0], "\n";
echo "GPU/fp128 clip(7,lo=10,hi=5): ", $rf2[1], "\n";

echo "DONE\n";
?>
--EXPECT--
GPU/float32 sign(NaN) is_nan: yes
GPU/float32 sign(+Inf): 1
GPU/float32 sign(-Inf): -1
GPU/float64 sign(NaN) is_nan: yes
GPU/float64 sign(+Inf): 1
GPU/float64 sign(-Inf): -1
GPU/float16 sign(NaN) is_nan: yes
GPU/float16 sign(+Inf): 1
GPU/float16 sign(-Inf): -1
GPU/fp128 sign(nan): nan
GPU/fp128 sign(0): 0
GPU/fp128 sign(-1): -1
GPU/float32 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
GPU/float32 clip(any, lo>hi)=hi: OK
GPU/float64 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
GPU/float64 clip(any, lo>hi)=hi: OK
GPU/float16 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
GPU/float16 clip(any, lo>hi)=hi: OK
GPU/fp128 clip(nan,0,5): nan
GPU/fp128 clip(3,0,5): 3
GPU/fp128 clip(100,0,5): 5
GPU/fp128 clip(3,lo=10,hi=5): 5
GPU/fp128 clip(7,lo=10,hi=5): 5
DONE
