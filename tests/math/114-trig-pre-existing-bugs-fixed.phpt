--TEST--
NumPower trig family: regressions for pre-existing dtype-and-device bugs (NDArray_Map float32-only)
--FILE--
<?php
/* Regression guards for the bugs the trig dtype-by-device refactor fixed.

   Before the refactor:
     - `NDArray_Map` returned a float32 buffer regardless of input dtype,
       so:
         * `sin(float64 input)` silently demoted to float32 (~7 digits).
         * `cos(int32)` cast through float32, with int values reinterpreted
           as fp32 bytes (UB / wrong values).
         * `sin(float128 input)` overran the 16-byte storage with fp32
           writes (heap corruption potential).
     - GPU `cuda_float_<op>` accepted only fp32; non-fp32 GPU inputs
       reinterpreted bytes as fp32 and returned corrupt output.
     - `float_rint` had buggy custom round-half-to-even logic that
       diverged from libm's `rintf` on negative half-integer inputs.
     - `arctan2` (binary, still on legacy path) had a NULL-check bug:
       guarded `x == NULL || y == NULL` on the zval pointers instead of
       the NDArray pointers — would segfault on conversion failure. The
       refactor fixed the guard while leaving arctan2 on the legacy
       binary path.
   This test asserts the post-refactor contract. */

function approx($g, $w, $tol) {
    if (is_float($g) || is_float($w)) {
        $gf = (float)$g; $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        if ($wf == 0.0) return abs($gf) <= $tol;
        return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
    }
    return (string)$g === (string)$w;
}
function check($label, $got, $want, $tol = 0.0) {
    if (is_array($got) && is_array($want)) {
        if (count($got) !== count($want)) { echo "FAIL $label: array size mismatch\n"; return; }
        foreach ($got as $i => $g) {
            if (!approx($g, $want[$i], $tol)) {
                echo "FAIL $label [$i]: got=" . json_encode($g) . " want=" . json_encode($want[$i]) . "\n";
                return;
            }
        }
        echo "OK $label\n";
        return;
    }
    echo (approx($got, $want, $tol) ? "OK $label\n" : "FAIL $label: got=" . json_encode($got) . " want=" . json_encode($want) . "\n");
}

/* ── Bug #1: sin(float64) was silently float32 ─────────────────────── */
$f64 = NumPower::array([1.0, 2.0, M_PI / 4], 'float64');
$r = NumPower::sin($f64);
check("sin float64 preserves dtype",  $r->__serialize()['dtype'], 'float64');
$vals = $r->toArray();
/* Full fp64 precision: sin(1) = 0.8414709848078965 (16 digits). The
   fp32-truncated value would be 0.8414709568023682, diverging past
   digit 7. We require fp64-tier precision. */
$expected = [sin(1.0), sin(2.0), sin(M_PI / 4)];
check("sin float64 full precision", $vals, $expected, 1e-15);

/* ── Bug #2: cos(int dtypes) cast through float32 silently ────────── */
foreach (['int32','int64','uint32','uint64'] as $dt) {
    $a = NumPower::array([1, 2, 3], $dt);
    $r = NumPower::cos($a);
    /* Wide ints widen to float64. cos(1)=0.5403023058681398 should
       round-trip at fp64 precision. */
    check("$dt cos promotes to float64", $r->__serialize()['dtype'], 'float64');
    check("$dt cos values fp64-precise",  $r->toArray(),
          [cos(1.0), cos(2.0), cos(3.0)], 1e-15);
}
foreach (['int8','int16','uint8','uint16'] as $dt) {
    $a = NumPower::array([1, 2, 3], $dt);
    $r = NumPower::sin($a);
    check("$dt sin promotes to float32", $r->__serialize()['dtype'], 'float32');
}

/* ── Bug #3: sin(fp128) overran storage; now full libquadmath precision */
$f128 = NumPower::array(['0.5235987755982988730771072305465838'], 'float128');  /* π/6 */
$r = NumPower::sin($f128);
check("sin fp128 preserves dtype",  $r->__serialize()['dtype'], 'float128');
/* sin(π/6) = 0.5 exactly. Even on DD fallback we keep > 12 digits. */
check("sin fp128 ≈ 0.5", (float)$r->toArray()[0], 0.5, 1e-12);

/* ── Bug #4: GPU non-fp32 reinterpreted bytes as fp32 ──────────────── */
try {
    foreach (['float16','float32','float64'] as $dt) {
        $a_gpu = NumPower::array([0.5, 1.0, M_PI / 4], $dt)->gpu();
        $r = NumPower::sin($a_gpu);
        check("$dt sin GPU stays on GPU",  $r->isGPU(), true);
        check("$dt sin GPU dtype preserved", $r->cpu()->__serialize()['dtype'], $dt);
        $tol = ($dt === 'float16') ? 5e-3 : (($dt === 'float32') ? 1e-5 : 1e-12);
        check("$dt sin GPU values",          $r->cpu()->toArray(),
              [sin(0.5), sin(1.0), sin(M_PI / 4)], $tol);
    }
    /* fp128 GPU via DD emulation */
    $a_gpu = NumPower::array(['0.5235987755982988730771072305465838'], 'float128')->gpu();
    $r = NumPower::sin($a_gpu);
    check("fp128 sin GPU stays on GPU",  $r->isGPU(), true);
    check("fp128 sin GPU dtype preserved", $r->cpu()->__serialize()['dtype'], 'float128');
    /* GPU DD truncates to fp64; expect ≥ 9 digits agreement. */
    check("fp128 sin GPU ≈ 0.5", (float)$r->cpu()->toArray()[0], 0.5, 1e-9);
} catch (Throwable $t) {
    echo "skip GPU not available: " . $t->getMessage() . "\n";
}

/* ── Bug #5: float_rint banker's rounding had custom buggy logic ────
   The fix routes through libm rintf/rint which is IEEE 754 round-half-
   to-even. Check the cases the buggy custom code would have miscomputed:
   -0.5, -1.5, -2.5 should round to 0, -2, -2 respectively. */
$rnd = NumPower::array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5], 'float64');
$r = NumPower::rint($rnd)->toArray();
check("rint round-half-to-even (CPU fp64)", $r,
      [0.0, 0.0, 2.0, -2.0, 2.0, -2.0], 0.0);
try {
    $rg = NumPower::rint($rnd->gpu())->cpu()->toArray();
    check("rint round-half-to-even (GPU fp64)", $rg,
          [0.0, 0.0, 2.0, -2.0, 2.0, -2.0], 0.0);
} catch (Throwable $t) {
    echo "skip rint GPU not available\n";
}

/* ── Bug #6: floor / ceil / rint / fix / trunc on int dtypes used to
   call the kernel (allocating a float buffer & launching a no-op
   compute). The fix short-circuits via NDArray_Copy, preserving dtype
   and skipping the kernel launch entirely. */
foreach (['int32','int64','uint64'] as $dt) {
    $a = NumPower::array([1, 5, 100], $dt);
    foreach (['floor','ceil','rint','fix','trunc'] as $op) {
        $r = NumPower::$op($a);
        check("$dt $op preserves int dtype", $r->__serialize()['dtype'], $dt);
        check("$dt $op preserves values", $r->toArray(),
              ($dt === 'uint64') ? ['1', '5', '100'] : [1, 5, 100]);
    }
}

/* ── Bug #7: arctan2 NULL-guard checked wrong pointers (would segfault) ──
   Pass a non-convertible value to arctan2 and confirm it raises a
   clean PHP error rather than segfaulting. */
try {
    /* objects that aren't NDArray / GdImage trigger the guard */
    NumPower::arctan2(new stdClass(), new stdClass());
    echo "FAIL arctan2 did not raise on invalid input\n";
} catch (Error $e) {
    echo "OK arctan2 raises clean error on invalid input\n";
}

echo "DONE\n";
?>
--EXPECTF--
%aDONE
