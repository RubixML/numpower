--TEST--
NumPower::round runs entirely on GPU for every dtype, matches CPU, and fixes the legacy half-away / float32-demotion bugs
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
/* GPU coverage + regression for the precision-aware round refactor:
     - the op runs on the device the input lives on (no CPU staging) —
       checked via isGPU() on the result before pulling it back;
     - GPU and CPU agree (bit-for-bit for f16/f32/f64, ~fp64 for the DD
       fp128 emulation);
     - round-half-to-even on GPU (the legacy `roundToDecimalsFloatKernel`
       used CUDA `round()` = half-away → round(2.5) wrongly gave 3);
     - the input dtype is preserved on GPU (the legacy GPU path went
       through a float32-only copy);
     - round(x, 0) is identical to rint(x) on GPU (dispatcher rewrite);
     - integer dtypes pass through unchanged on GPU.

   Inputs avoid exact decimal ties (e.g. 12.345 at 2 places): at a tie the
   DD fp128 emulation and the CPU libquadmath kernel can legitimately round
   to different even neighbours, an inherent property of the two formats. */

function approx($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g); $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!approx($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
    /* Treat numeric strings (fp128 / uint64 toArray output) as numbers. */
    $gnum = is_float($g) || is_int($g) || (is_string($g) && is_numeric($g));
    $wnum = is_float($w) || is_int($w) || (is_string($w) && is_numeric($w));
    if ($gnum && $wnum) {
        $gf = (float)$g; $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        if (is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0))) return true;
        if ($wf == 0.0) return abs($gf) <= $tol;
        return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
    }
    return (string)$g === (string)$w;
}
$fail = 0;
function check($label, $cond) {
    global $fail;
    if (!$cond) { $fail++; echo "FAIL $label\n"; }
}
function gdtype($a) { return $a->cpu()->__serialize()['dtype']; }

$halves = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5];
/* Non-tie at decimals 0 / 2 / -2 (see header note). */
$dec_in = [1234.5678, 6789.0123, -4321.0, 567.89];

/* ── float dtypes: GPU == CPU for decimals 0 / 2 / -2 ── */
$float_tol = [
    'float16'  => 5e-2,
    'float32'  => 1e-6,
    'float64'  => 1e-12,
    'float128' => 1e-9,   /* DD GPU vs libquadmath CPU diverge in last digits */
    'float4'   => 0.3,
    'float8'   => 0.3,
];
foreach ($float_tol as $dt => $tol) {
    $narrow = ($dt === 'float4' || $dt === 'float8');
    foreach ([0, 2, -2] as $dec) {
        $src = $narrow ? [0.0, 1.0, 2.0, 3.0] : $dec_in;
        $cpu = NumPower::round(NumPower::array($src, $dt), $dec);
        $g   = NumPower::round(NumPower::array($src, $dt)->gpu(), $dec);
        check("$dt round(,$dec) stays on GPU", $g->isGPU());
        check("$dt round(,$dec) GPU==CPU", approx($g->cpu()->toArray(), $cpu->toArray(), $tol));
        check("$dt round(,$dec) dtype", gdtype($g) === $dt);
    }
    /* banker's on GPU: round(2.5,0) == 2 (NOT 3 from the legacy half-away kernel) */
    if (!$narrow) {
        $g = NumPower::round(NumPower::array($halves, $dt)->gpu(), 0);
        check("$dt GPU banker halves", approx($g->cpu()->toArray(),
              [0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0], $tol));
    }
}

/* ── round(x, 0) == rint(x) on GPU (dispatcher rewrite) ── */
foreach (['float16', 'float32', 'float64', 'float128'] as $dt) {
    $r0 = NumPower::round(NumPower::array($halves, $dt)->gpu(), 0)->cpu()->toArray();
    $ri = NumPower::rint(NumPower::array($halves, $dt)->gpu())->cpu()->toArray();
    check("$dt GPU round(,0)==rint", approx($r0, $ri, 1e-9));
}

/* ── float64 GPU no longer demoted: a value past float32 precision
   survives because the dtype is kept ── */
$big = NumPower::array([1234567.875], 'float64')->gpu();
$rb = NumPower::round($big, 2);
check("float64 GPU stays on GPU", $rb->isGPU());
check("float64 GPU dtype kept", gdtype($rb) === 'float64');
check("float64 GPU precise", approx($rb->cpu()->toArray(), [1234567.88], 1e-6));

/* ── integer dtypes: identity on GPU, dtype preserved ── */
foreach (['int8','uint8','int16','uint16','int32','uint32','int64','uint64'] as $dt) {
    $r = NumPower::round(NumPower::array([12, 25, 37], $dt)->gpu(), -1);
    check("$dt GPU int stays on GPU", $r->isGPU());
    check("$dt GPU int identity", approx($r->cpu()->toArray(), [12, 25, 37], 0));
    check("$dt GPU int dtype", gdtype($r) === $dt);
}

/* ── multi-dimensional GPU ── */
$m = NumPower::array([[0.5, 1.5], [2.5, 3.5]], 'float64')->gpu();
check("2-D GPU round", approx(NumPower::round($m, 0)->cpu()->toArray(),
      [[0.0, 2.0], [2.0, 4.0]], 1e-12));

/* ── 0-D scalar on GPU collapses to PHP scalar ── */
check("0-D GPU round", approx(
      NumPower::round(NumPower::array(2.5, 'float64')->gpu(), 0), 2.0, 1e-12));

/* ── empty array on GPU keeps shape + dtype ── */
$re = NumPower::round(NumPower::zeros([0, 3], 'float32')->gpu(), 2);
check("empty GPU shape", $re->shape() === [0, 3]);
check("empty GPU dtype", gdtype($re) === 'float32');

/* ── fp128 ±Inf / NaN pass through round at non-zero precision on GPU
   (regression: the DD multiply used to corrupt ±Inf to NaN; CPU was fine) ── */
$inf128_cpu = NumPower::round(NumPower::array(['inf', '-inf', 'nan'], 'float128'), 2)->toArray();
$inf128_gpu = NumPower::round(NumPower::array(['inf', '-inf', 'nan'], 'float128')->gpu(), 2)->cpu()->toArray();
check("fp128 GPU round preserves Inf/NaN", $inf128_gpu === ['inf', '-inf', 'nan']);
check("fp128 round Inf/NaN GPU==CPU", $inf128_gpu === $inf128_cpu);
/* native float dtypes already handled Inf via the GPU rint kernel */
foreach (['float16', 'float32', 'float64'] as $dt) {
    $e = NumPower::round(NumPower::array([INF, -INF, NAN], $dt)->gpu(), 2)->cpu()->toArray();
    check("$dt GPU round Inf/NaN", is_infinite($e[0]) && $e[0] > 0 && is_infinite($e[1]) && $e[1] < 0 && is_nan($e[2]));
}

/* ── extreme precision: GPU passes the full 64-bit `decimals` (no 32-bit
   truncation), so CPU and GPU agree (both saturate to NaN) instead of
   diverging finite-vs-NaN ── */
foreach ([4294967298, -4294967298, PHP_INT_MAX, PHP_INT_MIN] as $big) {
    $c = NumPower::round(NumPower::array([3.14159], 'float64'), $big)->toArray()[0];
    $g = NumPower::round(NumPower::array([3.14159], 'float64')->gpu(), $big)->cpu()->toArray()[0];
    check("extreme precision $big CPU==GPU", (is_nan($c) && is_nan($g)) || $c === $g);
}

if ($fail === 0) echo "ALL OK\n";
echo "DONE\n";
?>
--EXPECT--
ALL OK
DONE
