--TEST--
NumPower::sum / prod with axis stay on GPU for every dtype, match the CPU path
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Axis-based sum / prod used to dispatch via NDArray_Add_Float (float32-
   only kernel) regardless of dtype, silently corrupting non-float32
   buffers and producing the wrong shape on GPU. The rewritten
   NDArray_Reduce_Axis routes per-dtype kernels through cuda_<op>_<tag>
   and casts the input to the widened dtype first so the accumulation
   stays on device for every supported dtype. */

$dtypes = [
    'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
];

$vals = [[1, 2, 3], [4, 5, 6]];
$want_sum0 = [5, 7, 9];     /* col sums: 1+4, 2+5, 3+6 */
$want_sum1 = [6, 15];       /* row sums: 1+2+3, 4+5+6 */
$want_prod0 = [4, 10, 18];  /* col prods */
$want_prod1 = [6, 120];     /* row prods */

$ok = true;
foreach ($dtypes as $t) {
    $a_cpu = NumPower::array($vals, $t);
    $a_gpu = $a_cpu->gpu();

    /* Sum axis=0 on GPU stays on GPU. */
    $r = NumPower::sum($a_gpu, 0);
    if (!$r->isGPU()) {
        echo "$t sum-axis0 not on GPU\n"; $ok = false;
    }
    $cpu_back = $r->cpu()->toArray();
    foreach ($want_sum0 as $i => $w) {
        if (abs((float)$cpu_back[$i] - $w) > 1e-5) {
            echo "$t sum-axis0 mismatch idx=$i got=", $cpu_back[$i], " want=$w\n";
            $ok = false;
        }
    }

    /* Sum axis=1 on GPU stays on GPU. */
    $r = NumPower::sum($a_gpu, 1);
    if (!$r->isGPU()) {
        echo "$t sum-axis1 not on GPU\n"; $ok = false;
    }
    $cpu_back = $r->cpu()->toArray();
    foreach ($want_sum1 as $i => $w) {
        if (abs((float)$cpu_back[$i] - $w) > 1e-5) {
            echo "$t sum-axis1 mismatch idx=$i got=", $cpu_back[$i], " want=$w\n";
            $ok = false;
        }
    }

    /* Prod axis=0 on GPU stays on GPU and matches CPU. */
    $r_gpu = NumPower::prod($a_gpu, 0)->cpu()->toArray();
    $r_cpu = NumPower::prod($a_cpu, 0)->toArray();
    foreach ($want_prod0 as $i => $w) {
        if (abs((float)$r_gpu[$i] - $w) > 1e-5 || abs((float)$r_cpu[$i] - $w) > 1e-5) {
            echo "$t prod-axis0 mismatch idx=$i gpu=", $r_gpu[$i], " cpu=", $r_cpu[$i], " want=$w\n";
            $ok = false;
        }
        /* CPU and GPU must agree exactly for values that fit in double. */
        if ((float)$r_gpu[$i] !== (float)$r_cpu[$i]) {
            echo "$t prod-axis0 CPU/GPU mismatch idx=$i gpu=", $r_gpu[$i], " cpu=", $r_cpu[$i], "\n";
            $ok = false;
        }
    }

    /* Prod axis=1 on GPU stays on GPU. */
    $r = NumPower::prod($a_gpu, 1);
    if (!$r->isGPU()) {
        echo "$t prod-axis1 not on GPU\n"; $ok = false;
    }
    $cpu_back = $r->cpu()->toArray();
    foreach ($want_prod1 as $i => $w) {
        if (abs((float)$cpu_back[$i] - $w) > 1e-5) {
            echo "$t prod-axis1 mismatch idx=$i got=", $cpu_back[$i], " want=$w\n";
            $ok = false;
        }
    }
}

/* Negative axis works on GPU too. */
$a = (new NDArray($vals, 'int32'))->gpu();
$r = NumPower::sum($a, -1);
if (!$r->isGPU() || $r->cpu()->toArray() !== $want_sum1) {
    echo "negative-axis GPU mismatch ", json_encode($r->cpu()->toArray()), "\n";
    $ok = false;
}

/* float4 / float8 GPU path stages through float16 — verify it works. */
foreach (['float4', 'float8'] as $t) {
    /* values exactly representable in fp4 / fp8 so the comparison stays sane */
    $a = (new NDArray([[1, 2], [2, 1]], $t))->gpu();
    $r = NumPower::sum($a, 0);
    if (!$r->isGPU()) {
        echo "$t axis=0 not on GPU\n"; $ok = false;
    }
    $back = $r->cpu()->toArray();
    if (abs((float)$back[0] - 3) > 1e-1 || abs((float)$back[1] - 3) > 1e-1) {
        echo "$t axis=0 mismatch ", json_encode($back), "\n";
        $ok = false;
    }
}

echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
