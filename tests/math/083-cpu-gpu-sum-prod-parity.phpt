--TEST--
NumPower::sum / prod give identical results on CPU and GPU for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The reductions are exposed on both devices; for every dtype the CPU
   and GPU paths must agree exactly when the sum / prod fits in the
   53-bit double accumulator used by the per-dtype GPU kernels. (For
   wider int64 / uint64 values past 2^53 the GPU kernel still goes through
   double — CPU then has the loss-free native accumulator and the two
   diverge intentionally; this test stays within the double-exact range.) */

$dtypes = [
    'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
];

$cases_1d = [
    [1, 2, 3, 4, 5],
    [0, 0, 0, 0, 0],
    [10, 20, 30],
];
$cases_2d = [
    [[1, 2, 3], [4, 5, 6]],
    [[2, 4], [3, 5], [7, 8]],
];

$ok = true;
foreach ($dtypes as $t) {
    foreach ($cases_1d as $c) {
        $a_cpu = new NDArray($c, $t);
        $a_gpu = $a_cpu->gpu();
        /* sum and prod with no axis. */
        $s_cpu = (float)NumPower::sum($a_cpu);
        $s_gpu = (float)NumPower::sum($a_gpu);
        $p_cpu = (float)NumPower::prod($a_cpu);
        $p_gpu = (float)NumPower::prod($a_gpu);
        /* float4 / float8 have very coarse quantisation; allow a small
           absolute tolerance for narrow floats. */
        $eps = ($t === 'float4' || $t === 'float8' || $t === 'float16') ? 1.0 : 1e-9;
        if (abs($s_cpu - $s_gpu) > $eps) {
            echo "$t 1D sum: CPU=$s_cpu GPU=$s_gpu\n"; $ok = false;
        }
        if (abs($p_cpu - $p_gpu) > $eps) {
            echo "$t 1D prod: CPU=$p_cpu GPU=$p_gpu\n"; $ok = false;
        }
    }
    foreach ($cases_2d as $c) {
        $a_cpu = new NDArray($c, $t);
        $a_gpu = $a_cpu->gpu();
        foreach ([0, 1] as $axis) {
            $r_cpu = NumPower::sum($a_cpu, $axis)->toArray();
            $r_gpu = NumPower::sum($a_gpu, $axis)->cpu()->toArray();
            foreach ($r_cpu as $i => $v) {
                $eps = ($t === 'float4' || $t === 'float8' || $t === 'float16') ? 1.0 : 1e-9;
                if (abs((float)$v - (float)$r_gpu[$i]) > $eps) {
                    echo "$t 2D sum axis=$axis idx=$i: CPU=", (string)$v,
                         " GPU=", (string)$r_gpu[$i], "\n";
                    $ok = false;
                }
            }
            $r_cpu = NumPower::prod($a_cpu, $axis)->toArray();
            $r_gpu = NumPower::prod($a_gpu, $axis)->cpu()->toArray();
            foreach ($r_cpu as $i => $v) {
                $eps = ($t === 'float4' || $t === 'float8' || $t === 'float16') ? 1.0 : 1e-9;
                if (abs((float)$v - (float)$r_gpu[$i]) > $eps) {
                    echo "$t 2D prod axis=$axis idx=$i: CPU=", (string)$v,
                         " GPU=", (string)$r_gpu[$i], "\n";
                    $ok = false;
                }
            }
        }
    }
}
echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
