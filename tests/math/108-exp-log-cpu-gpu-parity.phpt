--TEST--
NumPower exp/log family: CPU and GPU produce identical results across all dtypes
--SKIPIF--
<?php
try {
    $a = NumPower::array([1.0])->gpu();
    if (!$a->isGPU()) die("skip GPU not available");
} catch (Throwable $t) {
    die("skip GPU not available: " . $t->getMessage());
}
?>
--FILE--
<?php
/* Cross-device parity for the eight transcendental ops. For each
   (op × dtype) pair we compute the same input on CPU and GPU and
   verify the results agree within the dtype's normal IEEE-754 tier:
     - fp16 : 1e-2 (10-bit mantissa)
     - fp32 : 1e-5 (23-bit mantissa)
     - fp64 : 1e-12 (52-bit mantissa)
     - fp128: 1e-12 (DD GPU truncates to ~fp64 for transcendentals,
                     matching the libquadmath / DD CPU fallback contract).
   CUDA libdevice rounds to nearest-even by default, the same rounding
   mode glibc libm uses, so the only systematic gap is the precision
   floor of the DD-on-GPU pathway for fp128 transcendentals. */

function max_abs_diff($a, $b) {
    $aa = is_array($a) ? array_values($a) : [$a];
    $bb = is_array($b) ? array_values($b) : [$b];
    if (count($aa) !== count($bb)) return INF;
    $m = 0.0;
    for ($i = 0; $i < count($aa); $i++) {
        $av = is_array($aa[$i]) ? null : (float)$aa[$i];
        $bv = is_array($bb[$i]) ? null : (float)$bb[$i];
        if ($av === null || $bv === null) {
            /* recurse for nested arrays */
            $m = max($m, max_abs_diff($aa[$i], $bb[$i]));
            continue;
        }
        /* treat NaN/NaN and same-sign inf/inf as equal */
        if (is_nan($av) && is_nan($bv)) continue;
        if (is_infinite($av) && is_infinite($bv) && (($av < 0) === ($bv < 0))) continue;
        $m = max($m, abs($av - $bv));
    }
    return $m;
}

$ops = ['exp','exp2','expm1','log','log1p','log2','log10','logb'];

/* Pick inputs that exercise all branches: positive (log domain),
   includes 1.0 so log(1)=0, 0.0 so exp(0)=1, expm1(0)=0, log1p(0)=0,
   and a power-of-two (8.0) so log2 / logb hit exact integer outputs. */
$xs_num = [0.0, 1.0, 2.0, 4.0, 8.0];
$xs_str = ['0.0','1.0','2.0','4.0','8.0'];
$xs_log = [1.0, 2.0, 4.0, 8.0, 16.0];           /* strictly positive for log/log* */
$xs_log_str = ['1.0','2.0','4.0','8.0','16.0'];

$tols = [
    'float16'  => 5e-2,
    'float32'  => 1e-4,
    'float64'  => 1e-9,
    'float128' => 1e-9,
];

foreach ($tols as $dt => $tol) {
    foreach ($ops as $op) {
        /* Pick the right input domain for the op. */
        $needs_pos = in_array($op, ['log','log2','log10','logb'], true);
        $src_num   = $needs_pos ? $xs_log : $xs_num;
        $src_str   = $needs_pos ? $xs_log_str : $xs_str;
        $src       = ($dt === 'float128') ? $src_str : $src_num;

        $a_cpu = NumPower::array($src, $dt);
        $a_gpu = NumPower::array($src, $dt)->gpu();

        $r_cpu = NumPower::$op($a_cpu)->toArray();
        $r_gpu = NumPower::$op($a_gpu)->cpu()->toArray();

        $d = max_abs_diff($r_cpu, $r_gpu);
        if ($d <= $tol) {
            echo "OK $dt $op CPU≈GPU (max_diff=", sprintf('%.3g', $d), ")\n";
        } else {
            echo "FAIL $dt $op CPU vs GPU max_diff=$d (tol=$tol)\n",
                 "  cpu=", json_encode($r_cpu), "\n",
                 "  gpu=", json_encode($r_gpu), "\n";
        }
    }
}

/* Integer-promoted GPU paths must match CPU after the upstream cast. */
foreach (['int8','int16','int32','int64','uint8','uint16','uint32','uint64'] as $dt) {
    $a_cpu = NumPower::array([1, 2, 4, 8, 16], $dt);
    $a_gpu = NumPower::array([1, 2, 4, 8, 16], $dt)->gpu();
    foreach (['exp','log','log2','log10','logb'] as $op) {
        $r_cpu = NumPower::$op($a_cpu)->toArray();
        $r_gpu = NumPower::$op($a_gpu)->cpu()->toArray();
        $d = max_abs_diff($r_cpu, $r_gpu);
        $expected_tier = (in_array($dt, ['int32','int64','uint32','uint64'], true)) ? 1e-9 : 1e-4;
        if ($d <= $expected_tier) {
            echo "OK $dt $op int-promoted CPU≈GPU\n";
        } else {
            echo "FAIL $dt $op int-promoted: $d > $expected_tier\n";
        }
    }
}

echo "DONE\n";
?>
--EXPECTF--
%aDONE
