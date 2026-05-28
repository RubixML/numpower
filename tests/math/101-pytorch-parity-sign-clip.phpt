--TEST--
PyTorch parity for sign(NaN) and clamp/clip NaN-bounds / lo > hi semantics
--FILE--
<?php
/* PyTorch contract (Tensor methods):
   - torch.sign(NaN) returns NaN — IEEE-754-style propagation.
     Our prior `(x > 0) - (x < 0)` idiom returned 0 instead, which
     silently zero-ed out NaN inputs on every float dtype CPU + GPU.
   - torch.clamp(x, lo, hi) = std::min(std::max(x, lo), hi):
       • NaN in `x` propagates to the result;
       • NaN in `lo` or `hi` is swallowed (the value survives);
       • lo > hi → result is `hi`.
   This test covers every float dtype on both devices. */

$gpu_available = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_available = false; }

function approx($a, $b, $tol = 1e-5) {
    if (is_nan((float)$a) && is_nan((float)$b)) return true;
    return abs((float)$a - (float)$b) <= $tol;
}

/* ── sign(NaN) propagation ─────────────────────────────────────────────── */

foreach (['float32', 'float64', 'float16'] as $dt) {
    $arr  = NumPower::array([NAN, INF, -INF, 0.0, -3.0, 2.0], $dt);
    $sign = NumPower::sign($arr)->toArray();
    echo "CPU/$dt sign(NaN) is_nan: ", is_nan($sign[0]) ? "yes" : "no", "\n";
    echo "CPU/$dt sign(+Inf): ", $sign[1], "\n";
    echo "CPU/$dt sign(-Inf): ", $sign[2], "\n";
    echo "CPU/$dt sign(0): ",   $sign[3], "\n";
    echo "CPU/$dt sign(-3): ",  $sign[4], "\n";
    echo "CPU/$dt sign(2): ",   $sign[5], "\n";

    if ($gpu_available) {
        $g     = NumPower::array([NAN, INF, -INF, 0.0, -3.0, 2.0], $dt)->gpu();
        $sign  = NumPower::sign($g);
        if (!$sign->isGPU()) { echo "FAIL $dt sign left GPU\n"; continue; }
        $s_cpu = $sign->cpu()->toArray();
        echo "GPU/$dt sign(NaN) is_nan: ", is_nan($s_cpu[0]) ? "yes" : "no", "\n";
        echo "GPU/$dt sign(+Inf): ", $s_cpu[1], "\n";
        echo "GPU/$dt sign(-Inf): ", $s_cpu[2], "\n";
    }
}

/* ── fp128 sign(NaN) (DD on GPU; native __float128 on CPU libquadmath) ──── */

$f128       = NumPower::array(['nan', '0', '-1', '1e30'], 'float128');
$sign_f128  = NumPower::sign($f128)->toArray();
echo "CPU/fp128 sign(nan): ", $sign_f128[0], "\n";     /* expect "nan" */
echo "CPU/fp128 sign(0): ",   $sign_f128[1], "\n";     /* expect "0"   */
echo "CPU/fp128 sign(-1): ",  $sign_f128[2], "\n";     /* expect "-1"  */
echo "CPU/fp128 sign(1e30): ", $sign_f128[3], "\n";    /* expect "1"   */

if ($gpu_available) {
    $g128 = NumPower::array(['nan', '0', '-1', '1e30'], 'float128')->gpu();
    $sg   = NumPower::sign($g128);
    if (!$sg->isGPU()) { echo "FAIL fp128 sign left GPU\n"; }
    else {
        $a = $sg->cpu()->toArray();
        echo "GPU/fp128 sign(nan): ", $a[0], "\n";
        echo "GPU/fp128 sign(0): ",   $a[1], "\n";
        echo "GPU/fp128 sign(-1): ",  $a[2], "\n";
    }
}

/* ── clip / clamp semantics ────────────────────────────────────────────── */

/* lo > hi → return hi (PyTorch documented behaviour). */
foreach (['float32', 'float64', 'float16'] as $dt) {
    $arr = NumPower::array([-1.0, 3.0, 7.0, 100.0], $dt);
    $cl  = NumPower::clip($arr, 10.0, 5.0)->toArray();
    /* All elements end up clamped to hi=5. */
    $ok  = approx($cl[0], 5.0, 1e-3) && approx($cl[1], 5.0, 1e-3)
        && approx($cl[2], 5.0, 1e-3) && approx($cl[3], 5.0, 1e-3);
    echo "CPU/$dt clip(any, lo=10, hi=5)=hi: ", $ok ? "OK" : "FAIL " . json_encode($cl), "\n";
}

/* NaN-x propagates through clip → result is NaN. */
foreach (['float32', 'float64', 'float16'] as $dt) {
    $cl = NumPower::clip(NumPower::array([NAN, 3.0], $dt), 0.0, 5.0)->toArray();
    echo "CPU/$dt clip(NaN, 0, 5) is_nan: ", is_nan($cl[0]) ? "yes" : "no",
         "; clip(3, 0, 5)=", $cl[1], "\n";
}

/* NaN-bound is swallowed → result is x clamped against the finite bound. */
foreach (['float32', 'float64'] as $dt) {
    $cl1 = NumPower::clip(NumPower::array([3.0, 100.0], $dt), NAN, 5.0)->toArray();
    echo "CPU/$dt clip(x, NaN, 5): [3]=", $cl1[0], ", [100]=", $cl1[1], "\n";
    /* PyTorch: max(x, NaN) = x; min(x, 5). 3 stays 3, 100 clamped to 5. */

    $cl2 = NumPower::clip(NumPower::array([3.0, -100.0], $dt), 2.0, NAN)->toArray();
    echo "CPU/$dt clip(x, 2, NaN): [3]=", $cl2[0], ", [-100]=", $cl2[1], "\n";
    /* PyTorch: max(x, 2); min(_y, NaN) = _y. 3 stays 3, -100 clamped to 2. */
}

if ($gpu_available) {
    /* Replicate the key CPU checks on GPU and confirm they match. */
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

    /* fp128 clip on GPU. */
    $gf = NumPower::array(['nan', '3', '100'], 'float128')->gpu();
    $rf = NumPower::clip($gf, '0', '5')->cpu()->toArray();
    echo "GPU/fp128 clip(nan,0,5): ", $rf[0], "\n";
    echo "GPU/fp128 clip(3,0,5): ", $rf[1], "\n";
    echo "GPU/fp128 clip(100,0,5): ", $rf[2], "\n";

    $rf2 = NumPower::clip(NumPower::array(['3', '7'], 'float128')->gpu(),
                          '10', '5')->cpu()->toArray();
    echo "GPU/fp128 clip(3,lo=10,hi=5): ", $rf2[0], "\n";
    echo "GPU/fp128 clip(7,lo=10,hi=5): ", $rf2[1], "\n";
}

/* Integer clip lo > hi → also returns hi. */
foreach (['int8', 'int32', 'uint32', 'int64', 'uint64'] as $dt) {
    $vals = ($dt[0] === 'u') ? [1, 7, 100] : [-5, 7, 100];
    $r = NumPower::clip(NumPower::array($vals, $dt), 10, 5)->toArray();
    echo "CPU/$dt clip(any, lo=10, hi=5): ", json_encode($r), "\n";
}

echo "DONE\n";
?>
--EXPECTF--
CPU/float32 sign(NaN) is_nan: yes
CPU/float32 sign(+Inf): 1
CPU/float32 sign(-Inf): -1
CPU/float32 sign(0): 0
CPU/float32 sign(-3): -1
CPU/float32 sign(2): 1
%AGPU/float32 sign(NaN) is_nan: yes
GPU/float32 sign(+Inf): 1
GPU/float32 sign(-Inf): -1
%ACPU/float64 sign(NaN) is_nan: yes
CPU/float64 sign(+Inf): 1
CPU/float64 sign(-Inf): -1
CPU/float64 sign(0): 0
CPU/float64 sign(-3): -1
CPU/float64 sign(2): 1
%AGPU/float64 sign(NaN) is_nan: yes
GPU/float64 sign(+Inf): 1
GPU/float64 sign(-Inf): -1
%ACPU/float16 sign(NaN) is_nan: yes
CPU/float16 sign(+Inf): 1
CPU/float16 sign(-Inf): -1
CPU/float16 sign(0): 0
CPU/float16 sign(-3): -1
CPU/float16 sign(2): 1
%AGPU/float16 sign(NaN) is_nan: yes
GPU/float16 sign(+Inf): 1
GPU/float16 sign(-Inf): -1
CPU/fp128 sign(nan): nan
CPU/fp128 sign(0): 0
CPU/fp128 sign(-1): -1
CPU/fp128 sign(1e30): 1
%AGPU/fp128 sign(nan): nan
GPU/fp128 sign(0): 0
GPU/fp128 sign(-1): -1
CPU/float32 clip(any, lo=10, hi=5)=hi: OK
CPU/float64 clip(any, lo=10, hi=5)=hi: OK
CPU/float16 clip(any, lo=10, hi=5)=hi: OK
CPU/float32 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
CPU/float64 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
CPU/float16 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
CPU/float32 clip(x, NaN, 5): [3]=3, [100]=5
CPU/float32 clip(x, 2, NaN): [3]=3, [-100]=2
CPU/float64 clip(x, NaN, 5): [3]=3, [100]=5
CPU/float64 clip(x, 2, NaN): [3]=3, [-100]=2
%AGPU/float32 clip(NaN, 0, 5) is_nan: yes; clip(3, 0, 5)=3
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
CPU/int8 clip(any, lo=10, hi=5): [5,5,5]
CPU/int32 clip(any, lo=10, hi=5): [5,5,5]
CPU/uint32 clip(any, lo=10, hi=5): [5,5,5]
CPU/int64 clip(any, lo=10, hi=5): [5,5,5]
CPU/uint64 clip(any, lo=10, hi=5): ["5","5","5"]
DONE
