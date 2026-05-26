--TEST--
NumPower::truncatedNormal() produces CPU values that follow the truncated N(µ, σ²) | |z| ≤ 2
--FILE--
<?php
/* CPU-only statistical sanity for the truncated normal:
    - every accepted sample lies in `[loc - 2σ, loc + 2σ]` (the
      defining truncation property),
    - sample mean converges to µ (symmetric truncation),
    - sample stddev converges to 0.8796 σ (truncated variance factor).

   N = 16384 samples gives σ_eff/√N ≈ 0.007 for σ=1; a 0.20 tolerance
   on both moments fails loudly on any wiring bug yet passes with
   astronomical probability at the chosen N.

   The GPU distribution check has a dedicated counterpart in
   tests/initializers/091-numpower-truncated-normal-dtype-gpu.phpt
   (with --SKIPIF-- for non-CUDA builds), so this file deliberately
   does not exercise the GPU sampler — that keeps `--EXPECT--` strict
   on both CPU-only and CUDA-enabled CI matrices. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$n = 16384;

/* `NumPower::std` has a pre-existing dtype bug — read via toArray()
   on the dtype-correct path instead. */
function php_mean_std($a) {
    $arr = $a->toArray();
    $sum = 0.0; $cnt = 0;
    foreach ($arr as $v) { $sum += (float)$v; $cnt++; }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sq += $d * $d; }
    return [$mean, sqrt($sq / $cnt)];
}

function check_truncated($a, $mean_target, $std_target, $tag) {
    $arr = $a->toArray();
    $lo  = $mean_target - 2.0 * abs($std_target) - 1e-4;
    $hi  = $mean_target + 2.0 * abs($std_target) + 1e-4;
    $out = 0;
    foreach ($arr as $v) {
        $f = (float)$v;
        if ($f < $lo || $f > $hi) $out++;
    }
    [$m, $s] = php_mean_std($a);
    /* Truncated-normal stddev factor at ±2σ window is ~0.8796. */
    $truncated_std_target = 0.8796 * abs($std_target);
    $mean_ok = abs($m - $mean_target) < 0.20 * (abs($std_target) + 1.0);
    $std_ok  = abs($s - $truncated_std_target) <
               0.20 * (abs($truncated_std_target) + 1.0);
    echo $tag, ': out_of_window=', $out,
         ' mean=', ($mean_ok ? 'OK' : "BAD(t=$mean_target g=$m)"),
         ' std=',  ($std_ok  ? 'OK' : "BAD(t=$truncated_std_target g=$s)"), "\n";
}

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
?>
--EXPECT--
cpu_f32_standard: out_of_window=0 mean=OK std=OK
cpu_f32_shifted: out_of_window=0 mean=OK std=OK
cpu_f64_standard: out_of_window=0 mean=OK std=OK
cpu_i32_quantised: out_of_window=0 mean=OK std=OK
