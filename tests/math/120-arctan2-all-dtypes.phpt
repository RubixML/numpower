--TEST--
NumPower::arctan2 — all dtypes, broadcasting, promotion, string scalars, 0-D return, edge values (CPU)
--FILE--
<?php
/* Pure-CPU coverage for the arctan2 typed-binary refactor. Before it,
   arctan2 rode the legacy `NDArray_Map1ND` path which:
     - always allocated a float32 result (so arctan2(float64) truncated to
       fp32 garbage past ~7 digits, and float128 overran its storage);
     - iterated over numel(x) reading y[i] with NO broadcasting, so
       arctan2(array, scalar) read past the end of the scalar buffer.
   After: arctan2 rides `ndarray_promote_and_op` → per-dtype CPU kernels
   (`NDArray_Arctan2_{Float,Double,Float128}`), promoting both operands to a
   common float dtype and broadcasting like the arithmetic operators.

   Silent on success: only failures print a line, then a fixed summary. The
   strict --EXPECT-- (one summary line) means any value mismatch fails CI —
   unlike a `%a`-wildcard match. No GPU calls here, so this runs on every
   CI runner; CPU↔GPU parity is checked in 121. */

$FAILS = 0;
function near($g, $w, $tol) {
    $gf = (float)$g; $wf = (float)$w;
    if (is_nan($gf) && is_nan($wf)) return true;
    if (is_infinite($gf) && is_infinite($wf)) return ($gf < 0) === ($wf < 0);
    if ($wf == 0.0) return abs($gf) <= $tol;
    return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
}
function ok($cond, $label) {
    global $FAILS;
    if (!$cond) { echo "FAIL: $label\n"; $FAILS++; }
}
function dt($a) { return $a->__serialize()['dtype']; }

/* ── Signed float dtypes: full quadrant sweep vs PHP's libm atan2 ────────── */
/* arctan2(a, b) == C atan2(a, b): a is the numerator (y), b the denominator
   (x). The four sign combinations land in all four quadrants; the ±axis
   inputs exercise the (±π, ±π/2, 0) edges. */
$xa = [ 1.0, -1.0,  1.0, -1.0, 0.0,  0.0,  1.0, -1.0, 0.0];
$ya = [ 1.0,  1.0, -1.0, -1.0, 1.0, -1.0,  0.0,  0.0, 0.0];
$exp = [];
for ($i = 0; $i < count($xa); $i++) $exp[$i] = atan2($xa[$i], $ya[$i]);

$float_tol = ['float16' => 5e-3, 'float32' => 1e-6, 'float64' => 1e-14];
foreach ($float_tol as $t => $tol) {
    $r = NumPower::arctan2(NumPower::array($xa, $t), NumPower::array($ya, $t))->toArray();
    ok(dt(NumPower::arctan2(NumPower::array($xa, $t), NumPower::array($ya, $t))) === $t,
       "$t result dtype");
    for ($i = 0; $i < count($xa); $i++) {
        ok(near($r[$i], $exp[$i], $tol), "$t quadrant[$i] got={$r[$i]} want={$exp[$i]}");
    }
}

/* ── Signed integer dtypes: promote to float (narrow→f32, wide→f64) ──────── */
foreach (['int8' => 'float32', 'int16' => 'float32',
          'int32' => 'float64', 'int64' => 'float64'] as $t => $promo) {
    $xi = [1, -1, 1, -1, 0,  1, -1, 0];
    $yi = [1,  1, -1, -1, 1, 0,  0, 0];
    $res = NumPower::arctan2(NumPower::array($xi, $t), NumPower::array($yi, $t));
    ok(dt($res) === $promo, "$t promotes to $promo");
    $r = $res->toArray();
    $tol = ($promo === 'float64') ? 1e-14 : 1e-6;
    for ($i = 0; $i < count($xi); $i++) {
        ok(near($r[$i], atan2($xi[$i], $yi[$i]), $tol), "$t int quadrant[$i]");
    }
}

/* ── Unsigned integer dtypes: only non-negative inputs, result in [0, π/2] ─ */
foreach (['uint8' => 'float32', 'uint16' => 'float32',
          'uint32' => 'float64', 'uint64' => 'float64'] as $t => $promo) {
    $xu = [0, 1, 2, 3, 5];
    $yu = [1, 1, 0, 4, 0];
    $res = NumPower::arctan2(NumPower::array($xu, $t), NumPower::array($yu, $t));
    ok(dt($res) === $promo, "$t promotes to $promo");
    $r = $res->toArray();
    $tol = ($promo === 'float64') ? 1e-14 : 1e-6;
    for ($i = 0; $i < count($xu); $i++) {
        ok(near($r[$i], atan2($xu[$i], $yu[$i]), $tol), "$t uint quadrant[$i]");
    }
}

/* ── Narrow floats (float4 / float8): preserve dtype, exact at 0 ─────────── */
foreach (['float4', 'float8'] as $t) {
    $res = NumPower::arctan2(NumPower::array([0.0, 1.0], $t),
                            NumPower::array([1.0, 1.0], $t));
    ok(dt($res) === $t, "$t preserves dtype");
    $r = $res->toArray();
    ok(near($r[0], 0.0, 0.05), "$t arctan2(0,1)=0");   /* representable exactly */
}

/* ── float128: full libquadmath precision, native + string intake ───────── */
$qx = NumPower::array(['1.0', '1.0', '-1.0'], 'float128');
$qy = NumPower::array(['1.0', '0.0',  '1.0'], 'float128');
$rq = NumPower::arctan2($qx, $qy);
ok(dt($rq) === 'float128', 'fp128 result dtype');
$rqa = $rq->toArray();
$PI_4 = '0.78539816339744830961566084581988';   /* decimal reference, 32 digits */
/* Value checks use fp64 tolerance so they are PORTABLE: with libquadmath the
   fp128 atan2 is full 113-bit, but on the double-double fallback build
   (macOS / non-x86) fp128 *transcendentals* compute at fp64 precision (the
   NDARRAY_FP128_ATAN2 macro routes through atan2(double)). A >fp64 digit-prefix
   assertion would pass on Linux and FAIL on a DD build — so only assert it
   behind a runtime libquadmath probe below. */
ok(dt($rq) === 'float128', 'fp128 result dtype');
ok(near((float)$rqa[0],  M_PI / 4, 1e-13), 'fp128 atan2(1,1)=π/4 (fp64 tol)');
ok(near((float)$rqa[1],  M_PI / 2, 1e-13), 'fp128 atan2(1,0)=π/2 (fp64 tol)');
ok(near((float)$rqa[2], -M_PI / 4, 1e-13), 'fp128 atan2(-1,1)=-π/4 (fp64 tol)');

/* Probe for full-precision fp128 transcendentals (libquadmath): if the result
   already agrees with π/4 to 20 sig digits it cannot be the fp64 (~16-digit)
   DD result, so the build has libquadmath and we can assert the 30-digit
   prefix. On a DD build this probe is false and the strict check is skipped
   (the fp64-tolerance checks above already cover correctness there). */
$has_quadmath = (strncmp((string)$rqa[0], $PI_4, 20) === 0);
if ($has_quadmath) {
    ok(strncmp((string)$rqa[0], $PI_4, 30) === 0, 'fp128 atan2(1,1)=π/4 full precision (libquadmath)');
}

/* String scalar adopts the fp128 peer dtype (intake is loss-free on every
   build); the atan2 result value is checked at fp64 tolerance for portability. */
$rqs = NumPower::arctan2(NumPower::array(['1.0', '1.0'], 'float128'), '1.0');
ok(dt($rqs) === 'float128', 'fp128 + string scalar dtype');
ok(near((float)$rqs->toArray()[0], M_PI / 4, 1e-13), 'fp128 + string scalar value');

/* uint64 string intake keeps a > 2^53 magnitude loss-free in the denominator
   (result ~0 because the numerator 1 is tiny next to it, but the point is the
   intake does not corrupt the wide operand). */
$big = '18446744073709551615';                 /* UINT64_MAX */
$ru = NumPower::arctan2(NumPower::array(['1'], 'uint64'), $big);
ok(dt($ru) === 'float64', 'uint64 + string scalar promotes to float64');
ok(near($ru->toArray()[0], atan2(1.0, 1.8446744073709552e19), 1e-12), 'uint64 wide denominator');

/* ── Broadcasting (matches the arithmetic operators) ────────────────────── */
/* scalar literal + array */
$bs = NumPower::arctan2(NumPower::array([1.0, 2.0, 3.0], 'float64'), 1.0)->toArray();
ok(near($bs[0], atan2(1, 1), 1e-14) && near($bs[1], atan2(2, 1), 1e-14)
   && near($bs[2], atan2(3, 1), 1e-14), 'broadcast scalar denominator');
/* 0-D NDArray numerator + array */
$bn = NumPower::arctan2(NumPower::array(2.0, 'float64'),
                        NumPower::array([1.0, 2.0, 4.0], 'float64'))->toArray();
ok(near($bn[0], atan2(2, 1), 1e-14) && near($bn[2], atan2(2, 4), 1e-14), 'broadcast scalar numerator');
/* row vector (3,) broadcast across a (2,3) matrix */
$mat = NumPower::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 'float64');
$row = NumPower::array([1.0, 2.0, 3.0], 'float64');
$br = NumPower::arctan2($mat, $row)->toArray();
ok($br === [[atan2(1, 1), atan2(2, 2), atan2(3, 3)],
            [atan2(4, 1), atan2(5, 2), atan2(6, 3)]], 'broadcast row vector → matrix');
/* column vector (3,1) broadcast across a (3,3) matrix denominator */
$m33 = NumPower::array([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [1.0, 2.0, 4.0]], 'float64');
$col = NumPower::array([[1.0], [2.0], [3.0]], 'float64');
$bc = NumPower::arctan2($col, $m33)->toArray();
ok(near($bc[0][0], atan2(1, 1), 1e-14) && near($bc[1][2], atan2(2, 4), 1e-14)
   && near($bc[2][1], atan2(3, 2), 1e-14), 'broadcast column vector → matrix');

/* ── Larger array exceeding the small example dims (50 elements, 2 blocks) ─ */
$N = 50; $lx = []; $ly = [];
for ($i = 0; $i < $N; $i++) { $lx[$i] = ($i % 7) - 3.0; $ly[$i] = ($i % 5) - 2.0; }
$lr = NumPower::arctan2(NumPower::array($lx, 'float64'), NumPower::array($ly, 'float64'))->toArray();
$big_ok = true;
for ($i = 0; $i < $N; $i++) if (!near($lr[$i], atan2($lx[$i], $ly[$i]), 1e-14)) $big_ok = false;
ok($big_ok, 'large 1-D array element-wise');

/* ── 3-D shape ──────────────────────────────────────────────────────────── */
$c = NumPower::array([[[1.0, -1.0], [2.0, -2.0]]], 'float64');
$d = NumPower::array([[[1.0,  1.0], [0.0,  3.0]]], 'float64');
$cr = NumPower::arctan2($c, $d)->toArray();
ok($cr === [[[atan2(1, 1), atan2(-1, 1)], [atan2(2, 0), atan2(-2, 3)]]], '3-D element-wise');

/* ── 0-D return collapses to a PHP scalar ───────────────────────────────── */
$s = NumPower::arctan2(NumPower::array(1.0, 'float64'), NumPower::array(1.0, 'float64'));
ok(is_float($s) && near($s, M_PI / 4, 1e-14), '0-D float64 → PHP float');
/* Bare PHP scalars adopt the ZVAL_TO_NDARRAY default (float32), same as the
   arithmetic operators — still a PHP float, at float32 precision. */
$sbare = NumPower::arctan2(1.0, 1.0);
ok(is_float($sbare) && near($sbare, M_PI / 4, 1e-6), '0-D bare scalar → PHP float');
/* The important contract here is that a 0-D float128 result returns as a PHP
   *string* (not a lossy float); the value is checked at fp64 tolerance so the
   assertion is portable across the libquadmath and double-double builds. */
$sq = NumPower::arctan2(NumPower::array('1.0', 'float128'), NumPower::array('1.0', 'float128'));
ok(is_string($sq) && near((float)$sq, M_PI / 4, 1e-13), '0-D float128 → PHP string');

/* ── Edge values: ±inf, NaN, signed zero ────────────────────────────────── */
$ex = NumPower::array([INF,  INF, 1.0, NAN, 1.0], 'float64');
$ey = NumPower::array([INF, -INF, INF, 1.0, NAN], 'float64');
$er = NumPower::arctan2($ex, $ey)->toArray();
ok(near($er[0], M_PI / 4, 1e-14), 'atan2(inf,inf)=π/4');
ok(near($er[1], 3 * M_PI / 4, 1e-14), 'atan2(inf,-inf)=3π/4');
ok(near($er[2], 0.0, 1e-14), 'atan2(1,inf)=0');
ok(is_nan($er[3]), 'atan2(NaN,1)=NaN');
ok(is_nan($er[4]), 'atan2(1,NaN)=NaN');
/* Signed zero of the denominator splits +π vs -π. */
$zx = NumPower::array([0.0, -0.0], 'float64');
$zy = NumPower::array([-0.0, -0.0], 'float64');
$zr = NumPower::arctan2($zx, $zy)->toArray();
ok(near($zr[0],  M_PI, 1e-14), 'atan2(+0,-0)=+π');
ok(near($zr[1], -M_PI, 1e-14), 'atan2(-0,-0)=-π');

/* ── Legacy-bug guard: float64 not truncated to float32 ──────────────────── */
$prec_x = NumPower::array([0.123456789012345], 'float64');
$prec_y = NumPower::array([1.0], 'float64');
$pr = NumPower::arctan2($prec_x, $prec_y);
ok(dt($pr) === 'float64', 'precision guard: result is float64');
$want = atan2(0.123456789012345, 1.0);
/* A float32 round-trip would lose ~8 digits; require full f64 agreement. */
ok(abs((float)$pr->toArray()[0] - $want) < 1e-15, 'precision guard: full float64 value');

/* ── Empty arrays ───────────────────────────────────────────────────────── */
$e0 = NumPower::arctan2(NumPower::zeros([0, 4], 'float64'), NumPower::zeros([0, 4], 'float64'));
ok($e0->shape() === [0, 4] && dt($e0) === 'float64', 'empty (0,4) shape+dtype');

/* ── 0-D scalar broadcasts to a numel-1 array's shape (regression: CPU must
      not collapse to a 0-D scalar — it has to match the GPU / NumPy result
      shape (1,), see the DEFINE_ATAN2_FLOAT_CPU 0-D expansion) ────────────── */
$sc1 = NumPower::arctan2(NumPower::array(1.0, 'float64'), NumPower::array([2.0], 'float64'));
ok(is_object($sc1) && $sc1->shape() === [1], '0-D numerator + (1,) denominator -> shape (1,)');
ok(near($sc1->toArray()[0], atan2(1.0, 2.0), 1e-14), '0-D + (1,) value');
$sc2 = NumPower::arctan2(NumPower::array([2.0], 'float64'), NumPower::array(1.0, 'float64'));
ok(is_object($sc2) && $sc2->shape() === [1], '(1,) numerator + 0-D denominator -> shape (1,)');
/* float128 takes the same 0-D-expansion path */
$sc3 = NumPower::arctan2(NumPower::array('1.0', 'float128'), NumPower::array(['2.0'], 'float128'));
ok(is_object($sc3) && $sc3->shape() === [1], 'fp128 0-D + (1,) -> shape (1,)');
/* a genuine 0-D pair still collapses to a PHP scalar */
ok(is_float(NumPower::arctan2(NumPower::array(1.0, 'float64'), NumPower::array(1.0, 'float64'))),
   '0-D + 0-D -> PHP scalar');

/* ── String-operand validation: malformed literals throw, they are NOT
      silently coerced to 0 (regression: the binary dispatch must reject
      garbage like the unary path does) ─────────────────────────────────────── */
$peer = NumPower::array(['1.0'], 'float128');
foreach (['abc', '', '   ', '1.5.5', '0xff', '1,5'] as $bad) {
    $threw = false;
    try { NumPower::arctan2($peer, $bad); } catch (\Throwable $e) { $threw = true; }
    ok($threw, "arctan2(fp128, malformed '" . addslashes($bad) . "') throws");
}
/* valid numeric strings (incl. inf / nan / exponent / sign) are still accepted.
   Use a float64 peer so the 0-D result returns as a PHP float (PHP's
   (float)"nan" cast yields 0.0, not NAN — testing via the float64 toArray
   element is the reliable check). */
$peerf = NumPower::array([1.0], 'float64');
ok(near(NumPower::arctan2($peerf, 'inf')->toArray()[0], 0.0, 1e-30), "string 'inf' -> atan2(1,inf)=0");
ok(is_nan(NumPower::arctan2($peerf, 'nan')->toArray()[0]), "string 'nan' -> NaN");
ok(near(NumPower::arctan2($peerf, '-2')->toArray()[0], atan2(1.0, -2.0), 1e-12), "string '-2' accepted");
ok(near(NumPower::arctan2($peerf, '1e3')->toArray()[0], atan2(1.0, 1000.0), 1e-12), "string '1e3' accepted");

echo $FAILS === 0 ? "ALL CHECKS PASSED\n" : "TOTAL FAILURES: $FAILS\n";
?>
--EXPECT--
ALL CHECKS PASSED
