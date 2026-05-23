--TEST--
GPU linalg ops produce values matching the CPU path
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Pre-existing bug fix verification:
   - cuda_det_float used to return garbage (~1e-37) because it passed
     NULL workspace to cusolverDnSgetrf. After the workspace fix, GPU
     det must match CPU det within float32 tolerance.
   - cuda_matrix_float_inverse ignored cusolverDn errors and could
     produce undefined values; the fix adds status checks but the
     happy-path math must still match CPU. */

/* det */
$cases = [
    [[1.0, 2.0], [3.0, 4.0]],          /* -2 */
    [[2.0, 0.0], [0.0, 3.0]],          /* 6 */
    [[5.0]],                            /* 5 */
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  /* identity → 1 */
];
foreach ($cases as $i => $vals) {
    $m = new NDArray($vals, 'float32');
    $cpu = (float) NumPower::det($m);
    $gpu = (float) NumPower::det($m->gpu());
    if (abs($cpu - $gpu) > 1e-4 * (abs($cpu) > 1 ? abs($cpu) : 1)) {
        echo "det case $i: CPU=$cpu GPU=$gpu FAIL\n";
        exit;
    }
}
echo "det: ok\n";

/* inverse: A * inv(A) ≈ I on GPU. */
$m = (new NDArray([[4.0, 7.0], [2.0, 6.0]], 'float32'))->gpu();
$inv_gpu = NumPower::inv($m)->cpu()->toArray();
$cpu_m = new NDArray([[4.0, 7.0], [2.0, 6.0]], 'float32');
$inv_cpu = NumPower::inv($cpu_m)->toArray();
$ok = true;
for ($i = 0; $i < 2; $i++) {
    for ($j = 0; $j < 2; $j++) {
        if (abs($inv_gpu[$i][$j] - $inv_cpu[$i][$j]) > 1e-5) {
            $ok = false;
            echo "inv[$i][$j]: CPU=", $inv_cpu[$i][$j], " GPU=", $inv_gpu[$i][$j], "\n";
        }
    }
}
echo "inv: ", $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
det: ok
inv: ok
