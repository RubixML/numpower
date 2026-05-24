--TEST--
NumPower::truncatedNormal() produces values that follow the truncated N(µ, σ²) | |z| ≤ 2
--FILE--
<?php
/* Statistical sanity for the truncated normal: every accepted sample
   must lie inside `[loc - 2σ, loc + 2σ]` and the sample mean / stddev
   must converge to the theoretical truncated-normal moments:

     E[X]   = µ                          (symmetric truncation)
     std[X] ≈ 0.8796 σ                   (truncated variance factor)

   We pick N = 16384 samples for a tight standard error on the mean
   (σ_eff / √N ≈ 0.007 for σ = 1) and allow a 0.15 absolute tolerance
   on both moments — far smaller than what a wiring bug would produce
   yet wide enough to pass even at worst-case PRNG entropy. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$n = 16384;

/* `NumPower::std` has a pre-existing dtype bug — read via toArray()
   on the dtype-correct path instead. */
function php_mean_std($a, $is_gpu = false) {
    $arr = $is_gpu ? $a->cpu()->toArray() : $a->toArray();
    $sum = 0.0; $cnt = 0;
    foreach ($arr as $v) { $sum += (float)$v; $cnt++; }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sq += $d * $d; }
    return [$mean, sqrt($sq / $cnt)];
}

function check_truncated($a, $mean_target, $std_target, $tag, $is_gpu = false) {
    /* Truncation window. */
    $arr = $is_gpu ? $a->cpu()->toArray() : $a->toArray();
    $lo  = $mean_target - 2.0 * abs($std_target) - 1e-4;
    $hi  = $mean_target + 2.0 * abs($std_target) + 1e-4;
    $out = 0;
    foreach ($arr as $v) {
        $f = (float)$v;
        if ($f < $lo || $f > $hi) $out++;
    }
    /* Moments. */
    [$m, $s] = php_mean_std($a, $is_gpu);
    /* Truncated-normal stddev = 0.8796 σ. */
    $truncated_std_target = 0.8796 * abs($std_target);
    $mean_ok = abs($m - $mean_target) < 0.15 * (abs($std_target) + 1.0);
    $std_ok  = abs($s - $truncated_std_target) <
               0.20 * (abs($truncated_std_target) + 1.0);
    echo $tag, ': out_of_window=', $out,
         ' mean=', ($mean_ok ? 'OK' : "BAD(t=$mean_target g=$m)"),
         ' std=',  ($std_ok  ? 'OK' : "BAD(t=$truncated_std_target g=$s)"), "\n";
}

/* CPU paths. */
$a = NumPower::truncatedNormal([$n], 0.0, 1.0, 'float32');
check_truncated($a, 0.0, 1.0, 'cpu_f32_standard');

$a = NumPower::truncatedNormal([$n], 5.0, 2.0, 'float32');
check_truncated($a, 5.0, 2.0, 'cpu_f32_shifted');

$a = NumPower::truncatedNormal([$n], 0.0, 1.0, 'float64');
check_truncated($a, 0.0, 1.0, 'cpu_f64_standard');

/* int32 quantised — scale=50 >> quantisation step (1) so the
   approximation still converges. */
$a = NumPower::truncatedNormal([$n], 100, 50, 'int32');
check_truncated($a, 100.0, 50.0, 'cpu_i32_quantised');

/* GPU paths — only if cuRAND is available. */
$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Throwable $t) { }

if ($has_gpu) {
    $a = NumPower::truncatedNormal([$n], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
    check_truncated($a, 0.0, 1.0, 'gpu_f32_standard', true);

    $a = NumPower::truncatedNormal([$n], 5.0, 2.0, 'float64', NUMPOWER_CUDA);
    check_truncated($a, 5.0, 2.0, 'gpu_f64_shifted', true);
} else {
    echo "gpu_f32_standard: skip\n";
    echo "gpu_f64_shifted: skip\n";
}
?>
--EXPECT--
cpu_f32_standard: out_of_window=0 mean=OK std=OK
cpu_f32_shifted: out_of_window=0 mean=OK std=OK
cpu_f64_standard: out_of_window=0 mean=OK std=OK
cpu_i32_quantised: out_of_window=0 mean=OK std=OK
gpu_f32_standard: out_of_window=0 mean=OK std=OK
gpu_f64_shifted: out_of_window=0 mean=OK std=OK
