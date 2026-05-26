--TEST--
fp16 storage conversion matches GPU round-to-nearest (cuda __float2half) on CPU
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Pre-existing bug fixed: `ndarray_double_to_fp16` truncated the 13 low
   bits of the fp32 mantissa (round-toward-zero), while the GPU CUDA cast
   `__float2half` uses round-to-nearest-even. CPU and GPU diverged by up
   to one ULP for any fp16 value not exactly representable in fp32 input.
   The fix implements proper round-to-nearest-even on CPU to match GPU.

   This test exercises a handful of values where the old truncation
   diverged from the GPU and verifies CPU produces the same fp16 output. */

$samples = [
    /* 10.0 / 3.0 = 3.333... — closest fp16 is 3.333984375 (was 3.33203125). */
    [10.0, 3.0, 3.333984375],
    /* 1.0 / 7.0 = 0.142857... — closest fp16 is 0.14282... */
    [1.0, 7.0, 0.142822265625],
    /* pi ≈ 3.141592653 — closest fp16 is 3.140625 (round down: exact midpoint) */
    [M_PI * 2.0, 2.0, 3.140625],
    /* 1/3 ≈ 0.333... — closest fp16 is 0.333251953125 */
    [1.0, 3.0, 0.333251953125],
];

$ok = true;
foreach ($samples as $i => [$num, $den, $want]) {
    $a_cpu = new NDArray([$num], 'float16');
    $b_cpu = new NDArray([$den], 'float16');
    $r_cpu = (float)NumPower::divide($a_cpu, $b_cpu)[0];

    $a_gpu = $a_cpu->gpu();
    $b_gpu = $b_cpu->gpu();
    $r_gpu = (float)NumPower::divide($a_gpu, $b_gpu)->cpu()[0];

    /* fp16 has 11 mantissa bits → ~3-4 decimal digits of precision; allow a
       tiny absolute tolerance for the "want" comparison and require an exact
       match between CPU and GPU representations. */
    if ($r_cpu !== $r_gpu) {
        echo "sample $i CPU/GPU differ: CPU=$r_cpu GPU=$r_gpu\n";
        $ok = false;
    }
    if (abs($r_cpu - $want) > 1e-3) {
        echo "sample $i CPU mismatch: got=$r_cpu want=$want\n";
        $ok = false;
    }
}

/* Many sums + divides — confirm aggregate results match. */
$arr = [];
for ($i = 0; $i < 20; $i++) {
    $arr[] = 1.1 + $i * 0.123;
}
$cpu = new NDArray($arr, 'float16');
$gpu = $cpu->gpu();
if ((float)NumPower::sum($cpu) !== (float)NumPower::sum($gpu)) {
    echo "fp16 sum CPU=", NumPower::sum($cpu), " GPU=", NumPower::sum($gpu), "\n";
    $ok = false;
}

echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
