--TEST--
NumPower::randomBinomial() produces CPU values that follow the requested B(n, p)
--FILE--
<?php
/* CPU-only sanity checks on the sample distribution. For B(n, p),
   mean = np and variance = np(1-p). With N = 16384 samples and modest
   n, the standard error of the sample mean is sqrt(np(1-p)/N) ≈ 0.04
   for n=100, p=0.5 — so a 0.5 absolute tolerance is overwhelming
   probability.

   The GPU distribution check lives in
   tests/initializers/111-numpower-binomial-dtype-gpu.phpt (with
   --SKIPIF-- for non-CUDA builds), so this file deliberately stays on
   CPU only — keeps `--EXPECT--` strict on both CPU-only and
   CUDA-enabled CI matrices.

   Pre-existing bug pinned: the legacy `NDArray_Binomial` used
   `(float)rand() / (float)RAND_MAX` which could return exactly 1.0,
   causing the success test `u < p` to miss a success when `p == 1.0`.
   The new path uses `rand() / (RAND_MAX + 1.0)` so `p == 1.0` reliably
   yields every-sample-equals-n. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

$N = 16384;

function php_mean_var($a) {
    $arr = $a->toArray();
    $sum = 0.0; $cnt = 0;
    foreach ($arr as $v) { $sum += (float)$v; $cnt++; }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sq += $d * $d; }
    return [$mean, $sq / $cnt];
}

function check_binomial($a, $n, $p, $tag) {
    [$m, $v] = php_mean_var($a);
    $exp_mean = $n * $p;
    $exp_var  = $n * $p * (1.0 - $p);
    $mean_tol = max(0.5, 0.05 * ($exp_mean + 1.0));
    $var_tol  = max(0.5, 0.10 * ($exp_var + 1.0));
    $mean_ok = abs($m - $exp_mean) < $mean_tol;
    $var_ok  = abs($v - $exp_var)  < $var_tol;
    echo $tag, ': mean=', ($mean_ok ? 'OK' : "BAD(target=$exp_mean got=$m)"),
         ' var=', ($var_ok ? 'OK' : "BAD(target=$exp_var got=$v)"), "\n";
}

/* Bernoulli (n=1). */
$a = NumPower::randomBinomial([$N], 1, 0.5, 'float32');
check_binomial($a, 1, 0.5, 'cpu_B1_0.5');

/* Coin flips. */
$a = NumPower::randomBinomial([$N], 10, 0.5, 'float32');
check_binomial($a, 10, 0.5, 'cpu_B10_0.5');

/* Asymmetric p. */
$a = NumPower::randomBinomial([$N], 100, 0.3, 'int32');
check_binomial($a, 100, 0.3, 'cpu_B100_0.3');

/* High p — variance shrinks. */
$a = NumPower::randomBinomial([$N], 50, 0.9, 'float64');
check_binomial($a, 50, 0.9, 'cpu_B50_0.9');

/* Moderate n. */
$a = NumPower::randomBinomial([4096], 1000, 0.5, 'float64');
check_binomial($a, 1000, 0.5, 'cpu_B1000_0.5');

/* Counts must be non-negative integers in `[0, n]` regardless of dtype. */
$a = NumPower::randomBinomial([1024], 20, 0.5, 'float64');
$ok = true;
foreach ($a->toArray() as $v) {
    $f = (float)$v;
    if ($f < 0.0 || $f > 20.0 || $f !== floor($f)) { $ok = false; break; }
}
echo 'count_range: ', ($ok ? 'OK' : 'BAD'), "\n";

/* p = 1.0 every-sample-equals-n — pins the legacy 1.0-endpoint bug. */
$a = NumPower::randomBinomial([512], 25, 1.0, 'int32');
$all_n = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 25) { $all_n = false; break; }
}
echo 'p_eq_one_all_n: ', ($all_n ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
cpu_B1_0.5: mean=OK var=OK
cpu_B10_0.5: mean=OK var=OK
cpu_B100_0.3: mean=OK var=OK
cpu_B50_0.9: mean=OK var=OK
cpu_B1000_0.5: mean=OK var=OK
count_range: OK
p_eq_one_all_n: OK
