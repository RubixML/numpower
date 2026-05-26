--TEST--
NumPower::uniform() produces CPU values that follow the requested U([low, high))
--FILE--
<?php
/* CPU-only sanity checks on the sample distribution. N = 16384 samples
   give a standard error of (high - low) / sqrt(12 * N) ≈ 0.0023 for
   U([0, 1)), so a 0.05 tolerance on the mean passes with astronomical
   probability while still failing loudly if the path computes a
   distribution other than uniform.

   The GPU distribution check lives in
   tests/initializers/101-numpower-uniform-dtype-gpu.phpt (with
   --SKIPIF-- for non-CUDA builds), so this file deliberately stays on
   CPU only — keeps `--EXPECT--` strict on both CPU-only and
   CUDA-enabled CI matrices. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$n = 16384;

function php_mean($a) {
    $arr = $a->toArray();
    $sum = 0.0;
    $cnt = 0;
    foreach ($arr as $v) {
        $sum += (float)$v;
        $cnt++;
    }
    return $sum / $cnt;
}

function php_min_max($a) {
    $arr = $a->toArray();
    $min = INF; $max = -INF;
    foreach ($arr as $v) {
        $f = (float)$v;
        if ($f < $min) $min = $f;
        if ($f > $max) $max = $f;
    }
    return [$min, $max];
}

function check_dist($a, $low, $high, $tag) {
    $m = php_mean($a);
    [$mn, $mx] = php_min_max($a);
    $expected_mean = ($low + $high) / 2;
    $tol = 0.05 * abs($high - $low);
    $mean_ok = abs($m - $expected_mean) < $tol;
    /* Strict half-open interval: closed at low, open at high. */
    $range_ok = ($mn >= $low) && ($mx < $high);
    echo $tag, ': mean=', ($mean_ok ? 'OK' : "BAD(target=$expected_mean got=$m)"),
         ' range=', ($range_ok ? 'OK' : "BAD($mn..$mx vs [$low,$high))"), "\n";
}

$a = NumPower::uniform([$n], 0.0, 1.0, 'float32');
check_dist($a, 0.0, 1.0, 'cpu_f32_standard');

$a = NumPower::uniform([$n], -1.0, 1.0, 'float32');
check_dist($a, -1.0, 1.0, 'cpu_f32_symmetric');

$a = NumPower::uniform([$n], 0.0, 1.0, 'float64');
check_dist($a, 0.0, 1.0, 'cpu_f64_standard');

$a = NumPower::uniform([$n], 100.0, 200.0, 'float64');
check_dist($a, 100.0, 200.0, 'cpu_f64_shifted');

/* int32 over a wide range — the quantisation step (1) is far below
   the range (1000), so the sample moments still converge to the
   continuous uniform distribution. */
$a = NumPower::uniform([$n], 0, 1000, 'int32');
check_dist($a, 0, 1000, 'cpu_i32_quantised');
?>
--EXPECT--
cpu_f32_standard: mean=OK range=OK
cpu_f32_symmetric: mean=OK range=OK
cpu_f64_standard: mean=OK range=OK
cpu_f64_shifted: mean=OK range=OK
cpu_i32_quantised: mean=OK range=OK
