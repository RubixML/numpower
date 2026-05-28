--TEST--
sign(NaN) and clip NaN-bounds / lo > hi semantics (CPU)
--FILE--
<?php
/* Semantic contract:
   - sign(NaN) returns NaN — IEEE-754-style propagation.
     Our prior `(x > 0) - (x < 0)` idiom returned 0 instead, which
     silently zero-ed out NaN inputs on every float dtype CPU + GPU.
   - clip(x, lo, hi) = min(max(x, lo), hi):
       • NaN in `x` propagates to the result;
       • NaN in `lo` or `hi` is swallowed (the value survives);
       • lo > hi → result is `hi`.
   CPU half lives here; the GPU half is 102-sign-clip-gpu.phpt. */

function approx($a, $b, $tol = 1e-5) {
    if (is_nan((float)$a) && is_nan((float)$b)) return true;
    return abs((float)$a - (float)$b) <= $tol;
}

foreach (['float32', 'float64', 'float16'] as $dt) {
    $arr  = NumPower::array([NAN, INF, -INF, 0.0, -3.0, 2.0], $dt);
    $sign = NumPower::sign($arr)->toArray();
    echo "CPU/$dt sign(NaN) is_nan: ", is_nan($sign[0]) ? "yes" : "no", "\n";
    echo "CPU/$dt sign(+Inf): ", $sign[1], "\n";
    echo "CPU/$dt sign(-Inf): ", $sign[2], "\n";
    echo "CPU/$dt sign(0): ",   $sign[3], "\n";
    echo "CPU/$dt sign(-3): ",  $sign[4], "\n";
    echo "CPU/$dt sign(2): ",   $sign[5], "\n";
}

$f128       = NumPower::array(['nan', '0', '-1', '1e30'], 'float128');
$sign_f128  = NumPower::sign($f128)->toArray();
echo "CPU/fp128 sign(nan): ", $sign_f128[0], "\n";
echo "CPU/fp128 sign(0): ",   $sign_f128[1], "\n";
echo "CPU/fp128 sign(-1): ",  $sign_f128[2], "\n";
echo "CPU/fp128 sign(1e30): ", $sign_f128[3], "\n";

foreach (['float32', 'float64', 'float16'] as $dt) {
    $arr = NumPower::array([-1.0, 3.0, 7.0, 100.0], $dt);
    $cl  = NumPower::clip($arr, 10.0, 5.0)->toArray();
    $ok  = approx($cl[0], 5.0, 1e-3) && approx($cl[1], 5.0, 1e-3)
        && approx($cl[2], 5.0, 1e-3) && approx($cl[3], 5.0, 1e-3);
    echo "CPU/$dt clip(any, lo=10, hi=5)=hi: ", $ok ? "OK" : "FAIL " . json_encode($cl), "\n";
}

foreach (['float32', 'float64', 'float16'] as $dt) {
    $cl = NumPower::clip(NumPower::array([NAN, 3.0], $dt), 0.0, 5.0)->toArray();
    echo "CPU/$dt clip(NaN, 0, 5) is_nan: ", is_nan($cl[0]) ? "yes" : "no",
         "; clip(3, 0, 5)=", $cl[1], "\n";
}

foreach (['float32', 'float64'] as $dt) {
    $cl1 = NumPower::clip(NumPower::array([3.0, 100.0], $dt), NAN, 5.0)->toArray();
    echo "CPU/$dt clip(x, NaN, 5): [3]=", $cl1[0], ", [100]=", $cl1[1], "\n";

    $cl2 = NumPower::clip(NumPower::array([3.0, -100.0], $dt), 2.0, NAN)->toArray();
    echo "CPU/$dt clip(x, 2, NaN): [3]=", $cl2[0], ", [-100]=", $cl2[1], "\n";
}

foreach (['int8', 'int32', 'uint32', 'int64', 'uint64'] as $dt) {
    $vals = ($dt[0] === 'u') ? [1, 7, 100] : [-5, 7, 100];
    $r = NumPower::clip(NumPower::array($vals, $dt), 10, 5)->toArray();
    echo "CPU/$dt clip(any, lo=10, hi=5): ", json_encode($r), "\n";
}

echo "DONE\n";
?>
--EXPECT--
CPU/float32 sign(NaN) is_nan: yes
CPU/float32 sign(+Inf): 1
CPU/float32 sign(-Inf): -1
CPU/float32 sign(0): 0
CPU/float32 sign(-3): -1
CPU/float32 sign(2): 1
CPU/float64 sign(NaN) is_nan: yes
CPU/float64 sign(+Inf): 1
CPU/float64 sign(-Inf): -1
CPU/float64 sign(0): 0
CPU/float64 sign(-3): -1
CPU/float64 sign(2): 1
CPU/float16 sign(NaN) is_nan: yes
CPU/float16 sign(+Inf): 1
CPU/float16 sign(-Inf): -1
CPU/float16 sign(0): 0
CPU/float16 sign(-3): -1
CPU/float16 sign(2): 1
CPU/fp128 sign(nan): nan
CPU/fp128 sign(0): 0
CPU/fp128 sign(-1): -1
CPU/fp128 sign(1e30): 1
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
CPU/int8 clip(any, lo=10, hi=5): [5,5,5]
CPU/int32 clip(any, lo=10, hi=5): [5,5,5]
CPU/uint32 clip(any, lo=10, hi=5): [5,5,5]
CPU/int64 clip(any, lo=10, hi=5): [5,5,5]
CPU/uint64 clip(any, lo=10, hi=5): ["5","5","5"]
DONE
