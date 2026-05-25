--TEST--
NumPower::poisson() produces CPU values that follow the requested Poisson(λ)
--FILE--
<?php
/* CPU-only sanity checks on the sample distribution. For Poisson(λ),
   mean = variance = λ. With N = 16384 samples and modest λ, the
   standard error of the mean is sqrt(λ/N) ≈ 0.025 for λ=10, so a
   0.5 absolute tolerance passes with overwhelming probability while
   still failing loudly if the path computes a non-Poisson distribution.

   The GPU distribution check lives in
   tests/initializers/106-numpower-poisson-dtype-gpu.phpt (with
   --SKIPIF-- for non-CUDA builds), so this file stays on CPU only.

   Pre-existing bug pinned: legacy `NDArray_Poisson` used
   `expf(-lam)` which underflowed to 0 for λ ≥ 88 and caused the
   Knuth inner loop to spin forever. The new implementation uses
   `exp(-lam)` for λ < 30 and Hörmann's PTRS for λ ≥ 30 — both paths
   covered below. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$n = 16384;

function php_mean_var($a) {
    $arr = $a->toArray();
    $sum = 0.0; $cnt = 0;
    foreach ($arr as $v) { $sum += (float)$v; $cnt++; }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sq += $d * $d; }
    return [$mean, $sq / $cnt];
}

function check_poisson($a, $lam_target, $tag) {
    [$m, $v] = php_mean_var($a);
    /* Mean tolerance scales with the standard error sqrt(λ / N) so a
       given multiple-of-σ window is preserved across λ. Variance of
       the sample variance scales with sqrt(2)·λ/sqrt(N-1), so we use
       a separate tolerance proportional to sqrt(λ). Both are sized
       generously (~5σ) so the test passes essentially always while
       still failing loudly if the path computes wrong moments. */
    $n_samples = 16384;
    $mean_tol  = max(0.05, 5.0 * sqrt($lam_target / $n_samples));
    $var_tol   = max(0.10, 5.0 * sqrt(2.0) * $lam_target / sqrt($n_samples));
    $mean_ok = abs($m - $lam_target) < $mean_tol;
    $var_ok  = abs($v - $lam_target) < $var_tol;
    echo $tag, ': mean=', ($mean_ok ? 'OK' : "BAD(target=$lam_target got=$m)"),
         ' var=', ($var_ok ? 'OK' : "BAD(target=$lam_target got=$v)"), "\n";
}

/* Knuth branch: tiny lam, default lam, small lam. */
$a = NumPower::poisson([$n], 0.5, 'float32');
check_poisson($a, 0.5, 'cpu_knuth_lam0.5');

$a = NumPower::poisson([$n], 1.0, 'float32');
check_poisson($a, 1.0, 'cpu_knuth_lam1');

$a = NumPower::poisson([$n], 10.0, 'float64');
check_poisson($a, 10.0, 'cpu_knuth_lam10');

/* PTRS branch: lam >= 30. */
$a = NumPower::poisson([$n], 50.0, 'int32');
check_poisson($a, 50.0, 'cpu_ptrs_lam50');

$a = NumPower::poisson([$n], 1000.0, 'int64');
check_poisson($a, 1000.0, 'cpu_ptrs_lam1000');

/* Counts must be non-negative integers regardless of dtype. */
$a = NumPower::poisson([1024], 5.0, 'float64');
$nonneg_int = true;
foreach ($a->toArray() as $v) {
    $f = (float)$v;
    if ($f < 0.0 || $f !== floor($f)) { $nonneg_int = false; break; }
}
echo 'nonneg_integer_counts: ', ($nonneg_int ? 'OK' : 'BAD'), "\n";

/* Sanity: every sample for lam=0 is identically 0. */
$a = NumPower::poisson([100], 0.0, 'int32');
$all_zero = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'lam_zero_degenerate: ', ($all_zero ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
cpu_knuth_lam0.5: mean=OK var=OK
cpu_knuth_lam1: mean=OK var=OK
cpu_knuth_lam10: mean=OK var=OK
cpu_ptrs_lam50: mean=OK var=OK
cpu_ptrs_lam1000: mean=OK var=OK
nonneg_integer_counts: OK
lam_zero_degenerate: OK
