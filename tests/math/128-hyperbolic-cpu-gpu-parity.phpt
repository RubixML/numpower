--TEST--
NumPower hyperbolic family: CPU↔GPU parity across all dtypes incl. full-DD float128 (GPU)
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
/* CPU↔GPU parity for the six hyperbolic ops on every supported dtype.

   The headline check is float128: GPU has no native fp128, so it runs the
   double-double (DD) emulation. Previously the DD hyperbolic kernels
   collapsed to fp64 (dd_make(sinh(dd_to_double(x)), 0)), so a GPU fp128
   result agreed with the CPU libquadmath result only to ~16 digits. The
   DD path now composes the DD exp/log/sqrt primitives and agrees to ~30+
   significant digits — that regression is what `sig() >= 28` guards.

   Float / integer dtypes are compared with a per-dtype tolerance (last-ULP
   libm-vs-CUDA differences are expected). Gated on a present GPU by
   --SKIPIF--, so it is deterministic when it runs; CPU-only correctness is
   in 127. Silent on success → strict --EXPECT--. */

$FAILS = 0;
function ok($cond, $label) {
    global $FAILS;
    if (!$cond) { echo "FAIL: $label\n"; $FAILS++; }
}
function near($g, $w, $tol) {
    $gf = (float)$g; $wf = (float)$w;
    if (is_nan($gf) || is_nan($wf)) return is_nan($gf) && is_nan($wf);
    if (is_infinite($gf) || is_infinite($wf))
        return is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0));
    if ($wf == 0.0) return abs($gf) <= $tol;
    return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
}
function sig($a, $b) {
    if ($a === $b) return 99;
    $na = ltrim(str_replace(['-', '.'], '', $a), '0');
    $nb = ltrim(str_replace(['-', '.'], '', $b), '0');
    if ($na === '' && $nb === '') return 99;
    $n = min(strlen($na), strlen($nb));
    for ($i = 0; $i < $n; $i++) if ($na[$i] !== $nb[$i]) return $i;
    return $n;
}
$OPS = ['sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh'];

/* ── non-fp128 dtypes: CPU vs GPU element-wise within tolerance ─────────── */
/* Domain-aware inputs per op (arccosh ≥ 1, arctanh in (−1, 1)). */
$inp = [
    'sinh'    => [0.0, 0.0001, 0.5, 1.0, 2.0, -0.5, -1.5],
    'cosh'    => [0.0, 0.0001, 0.5, 1.0, 2.0, -0.5, -1.5],
    'tanh'    => [0.0, 0.0001, 0.5, 1.0, 3.0, -0.5, -1.0],
    'arcsinh' => [0.0, 0.0001, 0.5, 1.0, 5.0, -0.5, -2.0],
    'arccosh' => [1.0, 1.25, 2.0, 5.0],
    'arctanh' => [0.0, 0.0001, 0.5, 0.9, -0.5, -0.8],
];
$ftol = ['float16' => 3e-3, 'float32' => 3e-6, 'float64' => 1e-12,
         'int32' => 1e-12, 'uint32' => 1e-12, 'int64' => 1e-12, 'uint64' => 1e-12];
foreach ($ftol as $dt => $tol) {
    $isInt = $dt[0] === 'i' || $dt[0] === 'u';
    foreach ($OPS as $op) {
        foreach ($inp[$op] as $x) {
            if ($isInt && (float)(int)$x != $x) continue;   /* ints: integral inputs only */
            $val = $isInt ? (int)$x : $x;
            $cpu = NumPower::$op(new NDArray([$val], $dt))[0];
            $gpu = NumPower::$op((new NDArray([$val], $dt))->gpu())->cpu()[0];
            ok(near($cpu, $gpu, $tol), "$op($x) $dt CPU/GPU (cpu=$cpu gpu=$gpu)");
        }
    }
}

/* ── float128: CPU (libquadmath) vs GPU (DD) to ≥28 significant digits ──── */
$fp = [
    'sinh'    => ['0', '0.0001', '0.5', '1', '2.5', '-0.5', '-3'],
    'cosh'    => ['0', '0.0001', '0.5', '1', '2.5', '-0.5', '-3'],
    'tanh'    => ['0', '0.0001', '0.5', '1', '5', '-0.5', '-2'],
    'arcsinh' => ['0', '0.0001', '0.5', '1', '10', '-0.5', '-7'],
    'arccosh' => ['1', '1.0001', '1.5', '2', '50'],
    'arctanh' => ['0', '0.0001', '0.5', '0.9', '0.999', '-0.5', '-0.95'],
];
$worst = 99;
foreach ($fp as $op => $xs) {
    foreach ($xs as $x) {
        $cpu = (string) NumPower::$op(new NDArray([$x], 'float128'))[0];
        $gpu = (string) NumPower::$op((new NDArray([$x], 'float128'))->gpu())->cpu()[0];
        $s = sig($cpu, $gpu);
        if ($s < $worst) $worst = $s;
        ok($s >= 28, "fp128 $op($x) CPU/GPU agree<28 (got $s; cpu=$cpu gpu=$gpu)");
    }
}
/* fp128 special values must match exactly as rendered strings. */
$spec = [
    ['arccosh', '0.5', 'nan'], ['arccosh', '0', 'nan'],
    ['arctanh', '2',   'nan'], ['arctanh', '-2', 'nan'],
    ['arctanh', '1',   'inf'], ['arctanh', '-1', '-inf'],
];
foreach ($spec as [$op, $x, $want]) {
    $cpu = (string) NumPower::$op(new NDArray([$x], 'float128'))[0];
    $gpu = (string) NumPower::$op((new NDArray([$x], 'float128'))->gpu())->cpu()[0];
    ok($cpu === $want && $gpu === $want, "fp128 $op($x) special (cpu=$cpu gpu=$gpu want=$want)");
}

/* ── sinh/cosh overflow & ±inf regression guard (fp128) ─────────────────── */
/* dd_exp saturates to ±inf beyond ~709.78; the exp-difference/sum must NOT
   produce NaN. Below the saturation point the DD precision is preserved
   (≥28 digits); at the boundary (≤~710.48) the value is finite (fp64-tier);
   beyond, and for ±inf inputs, the DD result is correctly-signed inf — never
   NaN. (CPU libquadmath keeps finite values past 710.48 — that GPU-DD vs CPU
   divergence is the inherent fp64 exponent-range limit, shared with exp/log.) */
$gpu128 = fn($op, $x) => (string) NumPower::$op((new NDArray([$x], 'float128'))->gpu())->cpu()[0];
$cpu128 = fn($op, $x) => (string) NumPower::$op(new NDArray([$x], 'float128'))[0];
foreach (['sinh', 'cosh'] as $op) {
    foreach (['700', '-700', '709', '-709'] as $x) {        /* below cutoff: full DD */
        ok(sig($cpu128($op, $x), $gpu128($op, $x)) >= 28, "fp128 $op($x) DD parity ≥28");
    }
    foreach (['710', '-710'] as $x) {                       /* boundary: finite, fp64-tier */
        $c = $cpu128($op, $x); $g = $gpu128($op, $x);
        ok(stripos($g, 'nan') === false && stripos($g, 'inf') === false,
           "fp128 $op($x) finite (not nan/inf), got $g");
        ok(near((float)$c, (float)$g, 1e-12), "fp128 $op($x) CPU≈GPU fp64 tol");
    }
}
/* beyond DD range: correctly-signed inf, never NaN. */
ok($gpu128('sinh', '711')  === 'inf',  "fp128 sinh(711)→+inf not nan");
ok($gpu128('sinh', '-711') === '-inf', "fp128 sinh(-711)→-inf not nan");
ok($gpu128('sinh', '1000') === 'inf',  "fp128 sinh(1000)→+inf not nan");
ok($gpu128('cosh', '711')  === 'inf',  "fp128 cosh(711)→+inf not nan");
ok($gpu128('cosh', '-1000')=== 'inf',  "fp128 cosh(-1000)→+inf not nan");
/* ±inf inputs: CPU and GPU agree on the correctly-signed infinity. */
foreach ([['sinh', INF, 'inf'], ['sinh', -INF, '-inf'],
          ['cosh', INF, 'inf'], ['cosh', -INF, 'inf']] as [$op, $x, $want]) {
    $c = (string) NumPower::$op(new NDArray([$x], 'float128'))[0];
    $g = (string) NumPower::$op((new NDArray([$x], 'float128'))->gpu())->cpu()[0];
    ok($c === $want && $g === $want, "fp128 $op(±inf) CPU==GPU==$want (cpu=$c gpu=$g)");
}
/* fp64 too — sinh/cosh of ±inf and overflow must match CPU, never NaN. */
foreach ([['sinh', INF], ['sinh', -INF], ['cosh', INF], ['cosh', -INF],
          ['sinh', 710.0], ['cosh', 710.0], ['sinh', 1000.0]] as [$op, $x]) {
    $c = NumPower::$op(new NDArray([$x], 'float64'))[0];
    $g = NumPower::$op((new NDArray([$x], 'float64'))->gpu())->cpu()[0];
    ok(!is_nan((float)$g) && near($c, $g, 1e-12), "fp64 $op($x) CPU==GPU not nan");
}

/* ── multi-block (N > one CUDA block) and multi-dim parity ──────────────── */
$n = 4097;                                  /* forces >1 block on the GPU */
$vals = [];
for ($i = 0; $i < $n; $i++) $vals[] = (($i % 200) - 100) * 0.01;   /* in [-1, 1) */
$a64 = new NDArray($vals, 'float64');
foreach (['tanh', 'arcsinh'] as $op) {       /* both total over [-1,1) */
    $c = NumPower::$op($a64)->toArray();
    $g = NumPower::$op($a64->gpu())->cpu()->toArray();
    $md = 0.0;
    for ($i = 0; $i < $n; $i++) $md = max($md, abs($c[$i] - $g[$i]));
    ok($md < 1e-12, "$op multi-block N=$n CPU/GPU (maxdiff=$md)");
}
$mat = NumPower::reshape(new NDArray(range(1, 12), 'float64'), [3, 4]);  /* arccosh domain ≥1 */
$cm = NumPower::arccosh($mat)->toArray();
$gm = NumPower::arccosh($mat->gpu())->cpu()->toArray();
ok(NumPower::arccosh($mat->gpu())->shape() === [3, 4], "arccosh GPU keeps [3,4]");
$md = 0.0;
for ($i = 0; $i < 3; $i++) for ($j = 0; $j < 4; $j++) $md = max($md, abs($cm[$i][$j] - $gm[$i][$j]));
ok($md < 1e-12, "arccosh 2-D CPU/GPU (maxdiff=$md)");

/* ── narrow floats fp4/fp8/fp16 stay on GPU and match CPU ──────────────── */
foreach (['float4', 'float8', 'float16'] as $dt) {
    $c = (string) NumPower::tanh(new NDArray([0.5], $dt))[0];
    $g = (string) NumPower::tanh((new NDArray([0.5], $dt))->gpu())->cpu()[0];
    ok($c === $g, "tanh($dt) CPU==GPU (cpu=$c gpu=$g)");
}

echo $FAILS === 0 ? "DONE\n" : "$FAILS FAILURE(S) (fp128 worst sig=$worst)\n";
?>
--EXPECT--
DONE
