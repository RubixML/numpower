--TEST--
NumPower::normal() produces CPU values that follow the requested N(µ, σ²)
--FILE--
<?php
/* CPU-only sanity checks on the sample distribution. N = 16384 samples
   give a standard error of σ/√N ≈ 0.008 for σ = 1, so a 0.2σ tolerance
   on both mean and stddev passes with astronomical probability while
   still failing loudly if the path computes the wrong moments.

   NumPower::std has a pre-existing dtype bug on float64 (it always reads
   via NDArray_F32DATA), so we compute std manually from toArray() — that
   path is dtype-correct via the typed element decoder.

   The GPU distribution check has a dedicated counterpart in
   tests/initializers/086-numpower-normal-dtype-gpu.phpt (with --SKIPIF--
   for non-CUDA builds), so this file deliberately does not exercise the
   GPU sampler — that keeps `--EXPECT--` strict (no wildcards) on both
   CPU-only and CUDA-enabled CI matrices. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$n = 16384;

function php_mean_std($a) {
    $arr = $a->toArray();
    $sum = 0.0;
    $cnt = 0;
    foreach ($arr as $v) {
        $sum += (float)$v;
        $cnt++;
    }
    $mean = $sum / $cnt;
    $sq   = 0.0;
    foreach ($arr as $v) {
        $d   = ((float)$v) - $mean;
        $sq += $d * $d;
    }
    return [$mean, sqrt($sq / $cnt)];
}

function check_dist($a, $mean_target, $std_target, $tag) {
    [$m, $s] = php_mean_std($a);
    $tol = 0.20 * (abs($std_target) + 1.0);
    $mean_ok = abs($m - $mean_target) < $tol;
    $std_ok  = abs($s - $std_target)  < $tol;
    echo $tag, ': mean=', ($mean_ok ? 'OK' : "BAD(target=$mean_target got=$m)"),
         ' std=', ($std_ok ? 'OK' : "BAD(target=$std_target got=$s)"), "\n";
}

$a = NumPower::normal([$n], 0.0, 1.0, 'float32');
check_dist($a, 0.0, 1.0, 'cpu_f32_standard');

$a = NumPower::normal([$n], 5.0, 2.0, 'float32');
check_dist($a, 5.0, 2.0, 'cpu_f32_shifted');

$a = NumPower::normal([$n], 0.0, 1.0, 'float64');
check_dist($a, 0.0, 1.0, 'cpu_f64_standard');

/* int32 quantised — scale=50 >> quantisation step (1) so the moments
   still converge to the requested distribution. */
$a = NumPower::normal([$n], 100, 50, 'int32');
check_dist($a, 100.0, 50.0, 'cpu_i32_quantised');
?>
--EXPECT--
cpu_f32_standard: mean=OK std=OK
cpu_f32_shifted: mean=OK std=OK
cpu_f64_standard: mean=OK std=OK
cpu_i32_quantised: mean=OK std=OK
