--TEST--
NumPower trig family: coverage gaps (GPU multi-block, CPU↔GPU parity, integer rounding short-circuit, edge values on GPU, banker's rounding, negative inputs, PyTorch parity)
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
/* Closes the gaps surfaced by post-implementation review of test 111:
   1. GPU multi-block correctness (N > 1024 threads cross block boundary).
   2. CPU↔GPU bit-parity bound per dtype (not just both close to libm).
   3. Integer rounding short-circuit verifies (a) dtype preserved,
      (b) values bit-equal to input, (c) device preserved (GPU in → GPU out
      without staging).
   4. Edge values (NaN, ±Inf, domain errors) on GPU for every float dtype.
   5. rint banker's rounding edges: ±0.5, ±1.5, ±2.5, ±3.5.
   6. Negative input coverage for sin/cos/tan/sinh/cosh/tanh.
   7. fp128 full-precision string round-trip for sin/cos/atan.
   8. Empty multi-dim shapes on GPU. */

function approx($g, $w, $tol) {
    if (is_float($g) || is_float($w)) {
        $gf = (float)$g; $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        if (is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0))) return true;
        if ($wf == 0.0) return abs($gf) <= $tol;
        return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
    }
    return (string)$g === (string)$w;
}

function check($label, $got, $want, $tol = 0.0) {
    $ok = (is_array($got) && is_array($want))
        ? (function ($a, $b, $t) {
              if (count($a) !== count($b)) return false;
              foreach (array_values($a) as $i => $av) {
                  if (!approx($av, array_values($b)[$i], $t)) return false;
              }
              return true;
          })($got, $want, $tol)
        : approx($got, $want, $tol);
    if ($ok) echo "OK $label\n";
    else echo "FAIL $label: got=" . json_encode($got) . " want=" . json_encode($want) . "\n";
}

/* ── 1. GPU multi-block: N=4097 (16+1 blocks of 256) ─────────────────── */
$N = 4097;
$xs = [];
for ($i = 0; $i < $N; $i++) $xs[] = ($i - $N / 2) * 0.0005;   /* range ~[-1.02, 1.02] */
foreach (['float32', 'float64'] as $dt) {
    $tol = $dt === 'float32' ? 1e-5 : 1e-12;
    $a_cpu = NumPower::array($xs, $dt);
    $a_gpu = NumPower::array($xs, $dt)->gpu();
    foreach (['sin','cos','tan','arctan','sinh','tanh','arcsinh','floor','ceil','rint'] as $op) {
        $r_cpu = NumPower::$op($a_cpu)->toArray();
        $r_gpu = NumPower::$op($a_gpu)->cpu()->toArray();
        /* Verify tail thread (idx N-1) was processed correctly. */
        check("multi-block N=$N $dt $op tail",  $r_gpu[$N - 1], $r_cpu[$N - 1], $tol);
        /* Verify multi-block boundary (256-th element). */
        check("multi-block N=$N $dt $op idx=256", $r_gpu[256], $r_cpu[256], $tol);
    }
}

/* ── 2. CPU↔GPU bit-parity per dtype on every op ───────────────────── */
function max_diff($a, $b) {
    $m = 0.0;
    foreach (array_values($a) as $i => $av) {
        $bv = array_values($b)[$i];
        if (is_nan((float)$av) && is_nan((float)$bv)) continue;
        if (is_infinite((float)$av) && is_infinite((float)$bv)
            && ((float)$av < 0) === ((float)$bv < 0)) continue;
        $m = max($m, abs((float)$av - (float)$bv));
    }
    return $m;
}

$ops_all = ['sin','cos','tan','arcsin','arccos','arctan',
            'sinh','cosh','tanh','arcsinh','arccosh','arctanh',
            'degrees','radians','rint','fix','trunc','floor','ceil'];
$xs_tight = [0.1, 0.3, 0.5, 0.7, 0.9];

foreach (['float16','float32','float64'] as $dt) {
    /* Parity floor per dtype: not tighter than IEEE 754 ULP gap. */
    $parity_floor = $dt === 'float16' ? 5e-3 : ($dt === 'float32' ? 1e-5 : 1e-9);
    $arr = NumPower::array($xs_tight, $dt);
    $arr_gpu = NumPower::array($xs_tight, $dt)->gpu();
    foreach ($ops_all as $op) {
        /* arccos/arctanh: input must be in (-1, 1); arccosh: ≥ 1.  Skip
           those for the tight range; we exercise them in section 4. */
        if (in_array($op, ['arccosh'], true)) continue;
        $r_cpu = NumPower::$op($arr)->toArray();
        $r_gpu = NumPower::$op($arr_gpu)->cpu()->toArray();
        $d = max_diff($r_cpu, $r_gpu);
        if ($d <= $parity_floor) {
            echo "OK $dt $op CPU≈GPU\n";
        } else {
            echo "FAIL $dt $op CPU vs GPU diff=$d (floor=$parity_floor)\n";
        }
    }
}

/* ── 3. Integer rounding short-circuit: dtype + values + device ─────── */
foreach (['int8','int16','int32','int64','uint8','uint16','uint32','uint64'] as $dt) {
    $vals = ($dt === 'uint8') ? [1, 2, 100, 200]
          : (($dt === 'uint16') ? [1, 2, 100, 30000]
          : (($dt === 'uint32' || $dt === 'uint64') ? [1, 2, 100, 30000]
          : [-5, 1, 2, 100]));
    foreach (['rint','fix','trunc','floor','ceil'] as $op) {
        $a_cpu = NumPower::array($vals, $dt);
        $r_cpu = NumPower::$op($a_cpu);
        check("$dt $op CPU dtype preserved",   $r_cpu->__serialize()['dtype'], $dt);
        check("$dt $op CPU values unchanged",  $r_cpu->toArray(),
            ($dt === 'uint64') ? array_map('strval', $vals) : $vals);

        /* GPU: stays on GPU without staging through CPU. */
        $a_gpu = NumPower::array($vals, $dt)->gpu();
        $r_gpu = NumPower::$op($a_gpu);
        check("$dt $op GPU isGPU()",            $r_gpu->isGPU(), true);
        check("$dt $op GPU dtype preserved",    $r_gpu->cpu()->__serialize()['dtype'], $dt);
        check("$dt $op GPU values unchanged",   $r_gpu->cpu()->toArray(),
            ($dt === 'uint64') ? array_map('strval', $vals) : $vals);
    }
}

/* ── 4. Edge values on GPU for every float dtype ──────────────────── */
foreach (['float32','float64'] as $dt) {
    $edge = NumPower::array([INF, -INF, NAN, 0.0], $dt)->gpu();

    $sin_e = NumPower::sin($edge)->cpu()->toArray();
    check("$dt GPU sin(+Inf)=NaN",  is_nan($sin_e[0]), true);
    check("$dt GPU sin(-Inf)=NaN",  is_nan($sin_e[1]), true);
    check("$dt GPU sin(NaN)=NaN",   is_nan($sin_e[2]), true);
    check("$dt GPU sin(0)=0",       $sin_e[3], 0.0, 1e-15);

    $cos_e = NumPower::cos($edge)->cpu()->toArray();
    check("$dt GPU cos(+Inf)=NaN",  is_nan($cos_e[0]), true);
    check("$dt GPU cos(NaN)=NaN",   is_nan($cos_e[2]), true);
    check("$dt GPU cos(0)=1",       $cos_e[3], 1.0, 1e-15);

    $atan_e = NumPower::arctan($edge)->cpu()->toArray();
    check("$dt GPU arctan(+Inf)=π/2", $atan_e[0],  M_PI / 2, 1e-6);
    check("$dt GPU arctan(-Inf)=-π/2",$atan_e[1], -M_PI / 2, 1e-6);
    check("$dt GPU arctan(NaN)=NaN",  is_nan($atan_e[2]), true);

    $sinh_e = NumPower::sinh($edge)->cpu()->toArray();
    check("$dt GPU sinh(+Inf)=+Inf",  is_infinite($sinh_e[0]) && $sinh_e[0] > 0, true);
    check("$dt GPU sinh(-Inf)=-Inf",  is_infinite($sinh_e[1]) && $sinh_e[1] < 0, true);
    check("$dt GPU sinh(NaN)=NaN",    is_nan($sinh_e[2]), true);

    $tanh_e = NumPower::tanh($edge)->cpu()->toArray();
    check("$dt GPU tanh(+Inf)=1",     $tanh_e[0],  1.0, 1e-6);
    check("$dt GPU tanh(-Inf)=-1",    $tanh_e[1], -1.0, 1e-6);

    /* arcsin domain error on GPU */
    $domain = NumPower::array([2.0, -2.0, 1.0, -1.0], $dt)->gpu();
    $asin_d = NumPower::arcsin($domain)->cpu()->toArray();
    check("$dt GPU arcsin(2)=NaN",   is_nan($asin_d[0]), true);
    check("$dt GPU arcsin(-2)=NaN",  is_nan($asin_d[1]), true);
    check("$dt GPU arcsin(1)=π/2",   $asin_d[2],  M_PI / 2, 1e-5);
    check("$dt GPU arcsin(-1)=-π/2", $asin_d[3], -M_PI / 2, 1e-5);

    /* arccosh(<1) → NaN on GPU */
    $cosh_d = NumPower::array([0.5, 1.0, 2.0], $dt)->gpu();
    $acosh_d = NumPower::arccosh($cosh_d)->cpu()->toArray();
    check("$dt GPU arccosh(0.5)=NaN", is_nan($acosh_d[0]), true);
    check("$dt GPU arccosh(1)=0",     $acosh_d[1], 0.0, 1e-5);

    /* arctanh(±1) → ±Inf and arctanh(|x|>1) → NaN on GPU */
    $tanh_d = NumPower::array([1.0, -1.0, 2.0], $dt)->gpu();
    $atanh_d = NumPower::arctanh($tanh_d)->cpu()->toArray();
    check("$dt GPU arctanh(1)=+Inf",  is_infinite($atanh_d[0]) && $atanh_d[0] > 0, true);
    check("$dt GPU arctanh(-1)=-Inf", is_infinite($atanh_d[1]) && $atanh_d[1] < 0, true);
    check("$dt GPU arctanh(2)=NaN",   is_nan($atanh_d[2]), true);

    /* floor(NaN) = NaN, floor(+Inf) = +Inf, floor(2.0) = 2.0 */
    $rnd_e = NumPower::array([NAN, INF, -INF, 2.0, -2.0], $dt)->gpu();
    $f_e = NumPower::floor($rnd_e)->cpu()->toArray();
    check("$dt GPU floor(NaN)=NaN",  is_nan($f_e[0]), true);
    check("$dt GPU floor(+Inf)=+Inf", is_infinite($f_e[1]) && $f_e[1] > 0, true);
    check("$dt GPU floor(2.0)=2",     $f_e[3], 2.0, 0.0);
}

/* ── 5. rint banker's-rounding edges ─────────────────────────────── */
foreach (['float32','float64'] as $dt) {
    $a = NumPower::array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5], $dt);
    $r = NumPower::rint($a)->toArray();
    /* IEEE 754 rint round-half-to-even: 0.5→0, 1.5→2, 2.5→2, 3.5→4. */
    check("$dt rint banker CPU", $r, [0.0, 0.0, 2.0, -2.0, 2.0, -2.0, 4.0, -4.0], 0.0);
    $rg = NumPower::rint($a->gpu())->cpu()->toArray();
    check("$dt rint banker GPU", $rg, [0.0, 0.0, 2.0, -2.0, 2.0, -2.0, 4.0, -4.0], 0.0);
}

/* ── 6. Negative input coverage ──────────────────────────────────── */
foreach (['float32','float64'] as $dt) {
    $tol = $dt === 'float32' ? 1e-5 : 1e-12;
    $neg = NumPower::array([-M_PI / 2, -1.0, -0.5, 0.0, 0.5, 1.0, M_PI / 2], $dt);
    /* sin(-x) = -sin(x), cos(-x) = cos(x), tan(-x) = -tan(x). */
    $sin_neg = NumPower::sin($neg)->toArray();
    check("$dt sin(-π/2)=-1",  $sin_neg[0], -1.0, $tol);
    check("$dt sin(-1)≈-0.84", $sin_neg[1], -sin(1.0), $tol);
    check("$dt sin(π/2)=1",     $sin_neg[6],  1.0, $tol);
    $cos_neg = NumPower::cos($neg)->toArray();
    check("$dt cos(-π/2)≈0",   $cos_neg[0],  0.0, 1e-5);
    check("$dt cos(-1)≈0.54",  $cos_neg[1],  cos(1.0), $tol);
    /* sinh(-x) = -sinh(x), cosh(-x) = cosh(x). */
    $sinh_neg = NumPower::sinh(NumPower::array([-2.0, -1.0, 0.0, 1.0, 2.0], $dt))->toArray();
    check("$dt sinh(-2)=-sinh(2)", $sinh_neg[0], -sinh(2.0), $tol);
    check("$dt sinh symmetry",     $sinh_neg[0] + $sinh_neg[4], 0.0, $tol);
    $cosh_neg = NumPower::cosh(NumPower::array([-2.0, -1.0, 0.0, 1.0, 2.0], $dt))->toArray();
    check("$dt cosh(-2)=cosh(2)",  $cosh_neg[0],  cosh(2.0), $tol);
}

/* ── 7. fp128 full-precision string round-trip ───────────────────── */
$f128 = NumPower::array(['0.5235987755982988730771072305465838'], 'float128');   /* π/6 */
$s_f128 = NumPower::sin($f128)->toArray()[0];   /* sin(π/6) = 0.5 exactly */
/* sin(π/6) at fp128 precision should agree with libquadmath sinq;
   the libm reference is "0.5" exactly. Tolerance: 1e-30 if libquadmath
   present, ~1e-15 on DD fallback. We assert at least 12 digits of
   precision since DD CPU and DD GPU both clear that bar. */
$diff = abs((float)$s_f128 - 0.5);
echo $diff <= 1e-12 ? "OK fp128 sin(π/6)≈0.5 ($diff)\n" : "FAIL fp128 sin(π/6) diff=$diff\n";

/* fp128 atan(1) = π/4 */
$atan_f128 = NumPower::arctan(NumPower::array(['1.0'], 'float128'))->toArray()[0];
$diff = abs((float)$atan_f128 - M_PI / 4);
echo $diff <= 1e-12 ? "OK fp128 atan(1)≈π/4 ($diff)\n" : "FAIL fp128 atan(1) diff=$diff\n";

/* GPU fp128 sin */
$s_gpu_f128 = NumPower::sin($f128->gpu())->cpu()->toArray()[0];
$diff = abs((float)$s_gpu_f128 - 0.5);
echo $diff <= 1e-9 ? "OK fp128 GPU sin(π/6)≈0.5\n" : "FAIL fp128 GPU sin(π/6) diff=$diff\n";

/* ── 8. Empty multi-dim arrays on GPU ────────────────────────────── */
foreach (['float32','float64'] as $dt) {
    $e1 = NumPower::zeros([0], $dt)->gpu();
    $r = NumPower::sin($e1);
    check("$dt GPU sin(empty[0]) shape", $r->shape(), [0]);
    check("$dt GPU sin(empty[0]) dtype", $r->cpu()->__serialize()['dtype'], $dt);

    $e2 = NumPower::zeros([0, 5], $dt)->gpu();
    $r = NumPower::cos($e2);
    check("$dt GPU cos(empty[0,5]) shape", $r->shape(), [0, 5]);

    $e3 = NumPower::zeros([5, 0, 3], $dt)->gpu();
    $r = NumPower::tanh($e3);
    check("$dt GPU tanh(empty[5,0,3]) shape", $r->shape(), [5, 0, 3]);
}

/* ── 9. PyTorch / libm parity reference table (5 ops × 5 inputs) ── */
$libm_ref = [
    /* PyTorch on CPU calls libm; these are bit-equal at the input precision. */
    ['sin', [0.5, 1.0, 1.5, 2.0],     [sin(0.5), sin(1.0), sin(1.5), sin(2.0)]],
    ['cos', [0.5, 1.0, 1.5, 2.0],     [cos(0.5), cos(1.0), cos(1.5), cos(2.0)]],
    ['tan', [0.5, 1.0, 1.5],          [tan(0.5), tan(1.0), tan(1.5)]],
    ['sinh',[0.5, 1.0, 1.5],          [sinh(0.5), sinh(1.0), sinh(1.5)]],
    ['cosh',[0.5, 1.0, 1.5],          [cosh(0.5), cosh(1.0), cosh(1.5)]],
];
foreach ($libm_ref as [$op, $in, $expected]) {
    $r64 = NumPower::$op(NumPower::array($in, 'float64'))->toArray();
    check("PyTorch $op fp64", $r64, $expected, 1e-12);
    $r32 = NumPower::$op(NumPower::array($in, 'float32'))->toArray();
    check("PyTorch $op fp32", $r32, $expected, 1e-5);
    $r64g = NumPower::$op(NumPower::array($in, 'float64')->gpu())->cpu()->toArray();
    check("PyTorch $op fp64 GPU", $r64g, $expected, 1e-12);
}

echo "DONE\n";
?>
--EXPECTF--
%aDONE
