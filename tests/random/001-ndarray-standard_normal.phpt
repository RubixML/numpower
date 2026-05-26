--TEST--
NumPower::standardNormal() produces CPU values that follow N(0, 1)
--FILE--
<?php
/* CPU-only sanity checks on the standard-normal sample distribution.
   N = 16384 samples give a standard error of 1/sqrt(N) ≈ 0.008, so a
   0.2 absolute tolerance on mean and stddev passes with astronomical
   probability while still failing loudly if the path computes wrong
   moments.

   `NumPower::std` has a pre-existing dtype bug on float64 (it always
   reads via NDArray_F32DATA), so we compute std manually from toArray()
   — that path is dtype-correct via the typed element decoder.

   The GPU distribution check has a dedicated counterpart in
   tests/initializers/096-numpower-standard-normal-dtype-gpu.phpt (with
   --SKIPIF-- for non-CUDA builds), so this file deliberately does not
   exercise the GPU sampler — that keeps `--EXPECT--` strict (no
   wildcards) on both CPU-only and CUDA-enabled CI matrices. */

srand((int)(microtime(true) * 1000));
mt_srand((int)(microtime(true) * 1000));

function php_mean_std($a) {
    $arr = $a->toArray();
    $sum = 0.0;
    $cnt = 0;
    foreach ($arr as $v) {
        $sum += (float)$v;
        $cnt++;
    }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) {
        $d   = ((float)$v) - $mean;
        $sq += $d * $d;
    }
    return [$mean, sqrt($sq / $cnt)];
}

function check_dist($a, $tag) {
    [$m, $s] = php_mean_std($a);
    $mean_ok = abs($m) < 0.2;
    $std_ok  = abs($s - 1.0) < 0.2;
    echo $tag, ': mean=', ($mean_ok ? 'OK' : "BAD($m)"),
         ' std=', ($std_ok ? 'OK' : "BAD($s)"), "\n";
}

$n = 16384;

/* Default dtype (float32) and device (CPU). */
$a = NumPower::standardNormal([$n]);
check_dist($a, 'cpu_f32_default');

/* Explicit float32 / CPU. */
$a = NumPower::standardNormal([$n], 'float32');
check_dist($a, 'cpu_f32_explicit');

/* float64 / CPU — exercises the dtype-aware fill path. */
$a = NumPower::standardNormal([$n], 'float64');
check_dist($a, 'cpu_f64');

/* float16 / CPU — quantised path. f16 has ~3 decimal digits, so the
   sample moments still converge on N(0,1) to within the tolerance. */
$a = NumPower::standardNormal([$n], 'float16');
check_dist($a, 'cpu_f16_quantised');

/* int8 with implicit scale=1 — every sample is in (-128, 127). With
   loc=0 and scale=1 the truncation to integers means ~68% land at 0,
   so the mean stays near 0 and the stddev compresses below 1. We just
   spot-check the array shape and dtype rather than the moments. */
$a = NumPower::standardNormal([$n], 'int8');
$shape_ok = $a->shape() === [$n];
$type_ok  = gettype($a[0]) === 'integer';
echo 'cpu_i8: shape=', ($shape_ok ? 'OK' : 'BAD'),
     ' type=', ($type_ok ? 'OK' : 'BAD'), "\n";

/* 2-D multi-dim shape. */
$a = NumPower::standardNormal([128, 128]);
$ok = $a->shape() === [128, 128] && $a->size() === 16384;
echo 'cpu_2d: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
cpu_f32_default: mean=OK std=OK
cpu_f32_explicit: mean=OK std=OK
cpu_f64: mean=OK std=OK
cpu_f16_quantised: mean=OK std=OK
cpu_i8: shape=OK type=OK
cpu_2d: OK
