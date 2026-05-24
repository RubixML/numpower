--TEST--
NumPower::normal() produces values that follow the requested N(µ, σ²)
--FILE--
<?php
/* Sanity checks on the sample distribution. The test uses N = 16384
   samples, which gives a mean-of-mean standard error of σ / √N ≈ 0.008
   for σ = 1. We accept any |sample_mean - µ| < ~0.2 σ and
   |sample_std - σ| / σ < ~0.2, so the test passes with astronomical
   probability while still failing loudly if the path computes the wrong
   µ or σ.

   NumPower::std has a pre-existing dtype bug on float64 (it reads via
   NDArray_F32DATA regardless of dtype), so we compute std manually
   from toArray() — that path is dtype-correct via the typed-D2H route. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$n = 16384;

function php_mean_std($a, $is_gpu = false) {
    $arr = $is_gpu ? $a->cpu()->toArray() : $a->toArray();
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

function check_dist($a, $mean_target, $std_target, $tag, $is_gpu = false) {
    [$m, $s] = php_mean_std($a, $is_gpu);
    $mean_tol = 0.20 * (abs($std_target) + 1.0);
    $std_tol  = 0.20 * (abs($std_target) + 1.0);
    $mean_ok = abs($m - $mean_target) < $mean_tol;
    $std_ok  = abs($s - $std_target)  < $std_tol;
    echo $tag, ': mean=', ($mean_ok ? 'OK' : "BAD(target=$mean_target got=$m)"),
         ' std=', ($std_ok ? 'OK' : "BAD(target=$std_target got=$s)"), "\n";
}

/* CPU paths. */
$a = NumPower::normal([$n], 0.0, 1.0, 'float32');
check_dist($a, 0.0, 1.0, 'cpu_f32_standard');

$a = NumPower::normal([$n], 5.0, 2.0, 'float32');
check_dist($a, 5.0, 2.0, 'cpu_f32_shifted');

$a = NumPower::normal([$n], 0.0, 1.0, 'float64');
check_dist($a, 0.0, 1.0, 'cpu_f64_standard');

/* int32 quantised normal — scale=50 >> quantisation step (1) so mean
   and stddev still converge to the requested distribution. */
$a = NumPower::normal([$n], 100, 50, 'int32');
check_dist($a, 100.0, 50.0, 'cpu_i32_quantised');

/* GPU paths — only if cuRAND is available. */
$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Throwable $t) { }

if ($has_gpu) {
    $a = NumPower::normal([$n], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
    check_dist($a, 0.0, 1.0, 'gpu_f32_standard', true);

    $a = NumPower::normal([$n], 5.0, 2.0, 'float64', NUMPOWER_CUDA);
    check_dist($a, 5.0, 2.0, 'gpu_f64_shifted', true);
} else {
    echo "gpu_f32_standard: skip\n";
    echo "gpu_f64_shifted: skip\n";
}
?>
--EXPECT--
cpu_f32_standard: mean=OK std=OK
cpu_f32_shifted: mean=OK std=OK
cpu_f64_standard: mean=OK std=OK
cpu_i32_quantised: mean=OK std=OK
gpu_f32_standard: mean=OK std=OK
gpu_f64_shifted: mean=OK std=OK
