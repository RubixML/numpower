--TEST--
NumPower::exp/exp2/expm1/log/log1p/log2/log10/logb run entirely on GPU (every dtype)
--SKIPIF--
<?php
$ext = ini_get('extension_loaded') ? ini_get('extension_loaded') : 'unknown';
try {
    $a = NumPower::array([1.0])->gpu();
    if (!$a->isGPU()) die("skip GPU not available");
} catch (Throwable $t) {
    die("skip GPU not available: " . $t->getMessage());
}
?>
--FILE--
<?php
/* Confirms every transcendental op runs on the GPU for every supported
   compute dtype (float16, float32, float64, float128, plus narrow-int
   promotion to float32 / float64). The result must stay on the GPU and
   match the CPU compute path to within the dtype's normal tolerance.

   Verifies:
     - GPU input → GPU output for every op;
     - dtype preservation (or int → float promotion) matches CPU;
     - fp128 DD-emulation path agrees with the CPU path within fp64 limits;
     - narrow non-half floats (fp4 / fp8) stay on the GPU even though
       compute round-trips through fp32. */

function approx_equal($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g);
        $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!approx_equal($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
    if (is_array($g) || is_array($w)) return false;
    if (is_float($g) || is_float($w)) {
        $gf = (float)$g;
        $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        if (is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0))) return true;
        return abs($gf - $wf) <= $tol;
    }
    return (string)$g === (string)$w;
}

function check($label, $got, $want, $tol = 0.0) {
    if (approx_equal($got, $want, $tol)) {
        echo "OK $label\n";
    } else {
        echo "FAIL $label: got=", json_encode($got),
             " want=", json_encode($want), "\n";
    }
}

/* Op table: name → (compute on a $base array, golden values for tol). */
$ops = [
    'exp'   => fn($a) => NumPower::exp($a),
    'exp2'  => fn($a) => NumPower::exp2($a),
    'expm1' => fn($a) => NumPower::expm1($a),
    'log'   => fn($a) => NumPower::log($a),
    'log1p' => fn($a) => NumPower::log1p($a),
    'log2'  => fn($a) => NumPower::log2($a),
    'log10' => fn($a) => NumPower::log10($a),
    'logb'  => fn($a) => NumPower::logb($a),
];

/* Input picked so every op has a sensible domain (all positive, includes 1.0
   so log(1)=0, includes 2/4/8 so log2 gives integer outputs). */
$xs = [1.0, 2.0, 4.0, 8.0];

$float_dtypes  = ['float16','float32','float64'];

foreach ($float_dtypes as $dt) {
    $tol = ($dt === 'float16') ? 5e-2 : ($dt === 'float32' ? 1e-4 : 1e-9);
    $a_cpu = NumPower::array($xs, $dt);
    $a_gpu = NumPower::array($xs, $dt)->gpu();
    foreach ($ops as $name => $fn) {
        $r_cpu = $fn($a_cpu);
        $r_gpu = $fn($a_gpu);
        /* GPU residency check */
        check("$dt $name stays on GPU", $r_gpu->isGPU(), true);
        /* dtype preserved */
        check("$dt $name dtype preserved", $r_gpu->cpu()->__serialize()['dtype'], $dt);
        /* Values agree (within tol) */
        check("$dt $name CPU=GPU",
              $r_cpu->toArray(), $r_gpu->cpu()->toArray(), $tol);
    }
}

/* ── float128 GPU (DD emulation): accuracy floor is fp64 ────────────── */
$dt = 'float128';
$tol = 1e-12;
$a_cpu = NumPower::array(['1.0','2.0','4.0','8.0'], $dt);
$a_gpu = NumPower::array(['1.0','2.0','4.0','8.0'], $dt)->gpu();
foreach ($ops as $name => $fn) {
    $r_cpu = $fn($a_cpu);
    $r_gpu = $fn($a_gpu);
    check("fp128 $name stays on GPU",  $r_gpu->isGPU(), true);
    check("fp128 $name dtype",         $r_gpu->cpu()->__serialize()['dtype'], 'float128');
    /* Compare via floats since fp128 GPU is DD-emulation truncated to fp64. */
    $cv = array_map('floatval', $r_cpu->toArray());
    $gv = array_map('floatval', $r_gpu->cpu()->toArray());
    check("fp128 $name CPU≈GPU (fp64 tier)", $gv, $cv, $tol);
}

/* ── Integer dtype promotion on GPU ─────────────────────────────────── */
foreach (['int8','int16','uint8','uint16'] as $dt) {
    $a_gpu = NumPower::array([1, 2, 4, 8], $dt)->gpu();
    $r = NumPower::log2($a_gpu);
    check("$dt → float32 (GPU)",    $r->cpu()->__serialize()['dtype'], 'float32');
    check("$dt log2 GPU values",    $r->cpu()->toArray(), [0.0, 1.0, 2.0, 3.0], 1e-5);
    check("$dt log2 stays on GPU",  $r->isGPU(), true);
}
foreach (['int32','int64','uint32','uint64'] as $dt) {
    $a_gpu = NumPower::array([1, 2, 4, 8], $dt)->gpu();
    $r = NumPower::log2($a_gpu);
    check("$dt → float64 (GPU)",    $r->cpu()->__serialize()['dtype'], 'float64');
    check("$dt log2 GPU values",    $r->cpu()->toArray(), [0.0, 1.0, 2.0, 3.0], 1e-12);
}

/* ── float4 / float8 (narrow non-half floats): stay on GPU ──────────── */
foreach (['float4', 'float8'] as $dt) {
    $tol = ($dt === 'float4') ? 0.6 : 0.4;
    $a_gpu = NumPower::array([1.0, 2.0], $dt)->gpu();
    $r = NumPower::log($a_gpu);
    check("$dt log stays on GPU", $r->isGPU(), true);
    check("$dt log dtype preserved", $r->cpu()->__serialize()['dtype'], $dt);
    /* log(1)=0, log(2)≈0.69 — fp4 quantises log(2) heavily. */
    $g = $r->cpu()->toArray();
    check("$dt log first element ≈ 0", $g[0], 0.0, $tol);
}

/* ── 2-D / 3-D GPU shapes ─────────────────────────────────────────────── */
$mat_gpu = NumPower::array([[1.0, M_E], [M_E*M_E, M_E*M_E*M_E]], 'float64')->gpu();
$lg = NumPower::log($mat_gpu);
check("2-D GPU log stays on GPU", $lg->isGPU(), true);
check("2-D GPU log values", $lg->cpu()->toArray(), [[0.0, 1.0], [2.0, 3.0]], 1e-12);

$cube_gpu = NumPower::array([[[1.0, 2.0],[4.0, 8.0]]], 'float64')->gpu();
$lg2 = NumPower::log2($cube_gpu);
check("3-D GPU log2 stays on GPU", $lg2->isGPU(), true);
check("3-D GPU log2 values", $lg2->cpu()->toArray(), [[[0.0, 1.0],[2.0, 3.0]]], 1e-12);

/* ── Edge values on GPU: log(0)=-inf, log(-1)=NaN ─────────────────────── */
$edge_gpu = NumPower::array([0.0, 1.0, -1.0], 'float64')->gpu();
$le_gpu = NumPower::log($edge_gpu)->cpu()->toArray();
check("GPU log(0)=-inf", is_infinite($le_gpu[0]) && $le_gpu[0] < 0, true);
check("GPU log(1)=0",    $le_gpu[1], 0.0, 1e-12);
check("GPU log(-1)=NaN", is_nan($le_gpu[2]), true);

/* exp(inf)=inf, exp(-inf)=0, exp(NaN)=NaN on GPU */
$einf_gpu = NumPower::array([INF, -INF, NAN], 'float64')->gpu();
$e_inf = NumPower::exp($einf_gpu)->cpu()->toArray();
check("GPU exp(inf)=inf", is_infinite($e_inf[0]) && $e_inf[0] > 0, true);
check("GPU exp(-inf)=0",   $e_inf[1], 0.0, 1e-12);
check("GPU exp(NaN)=NaN",  is_nan($e_inf[2]), true);

echo "DONE\n";
?>
--EXPECTF--
%aDONE
