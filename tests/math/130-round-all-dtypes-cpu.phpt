--TEST--
NumPower::round across every dtype (CPU): banker's rounding, precision, dtype preservation, string intake
--FILE--
<?php
/* Comprehensive CPU coverage for the precision-aware `NumPower::round`
   refactor (formerly on the legacy float32-only `NDArray_Map1F` /
   `roundf` path). Verifies:
     - round-half-to-even (banker's rounding), matching PyTorch / NumPy:
       round(2.5) == 2, round(0.5) == 0, NOT the legacy half-away result;
     - the input dtype is preserved (no silent demotion to float32);
     - positive, zero and negative `precision` (decimals);
     - `precision` defaults to 0;
     - integer dtypes pass through unchanged (rounding family convention);
     - multi-dimensional shapes, 0-D scalars and empty arrays;
     - bare numeric-string intake (float128 / uint64 single-call precision);
     - boundary values (NaN / Inf, dtype extremes). */

function approx($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g); $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!approx($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
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
    echo (approx($got, $want, $tol) ? "OK " : "FAIL ") . $label;
    if (!approx($got, $want, $tol)) {
        echo ": got=", json_encode($got), " want=", json_encode($want);
    }
    echo "\n";
}
function dt_of($a) { return $a->__serialize()['dtype']; }

$tols = ['float16' => 5e-2, 'float32' => 1e-5, 'float64' => 1e-12, 'float128' => 1e-12];

/* ── banker's rounding (decimals = 0) + dtype preservation ── */
$halves_in  = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5];
$halves_out = [0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0];
foreach ($tols as $dt => $tol) {
    $a = NumPower::array($halves_in, $dt);
    $r = NumPower::round($a, 0);
    check("$dt halves->even", $r->toArray(), $halves_out, $tol);
    check("$dt dtype preserved", dt_of($r), $dt);
    /* precision defaults to 0 */
    check("$dt default precision", NumPower::round($a)->toArray(), $halves_out, $tol);
}

/* ── positive precision ── */
foreach ($tols as $dt => $tol) {
    $a = NumPower::array([3.14159, 2.71828], $dt);
    check("$dt round(,2)", NumPower::round($a, 2)->toArray(), [3.14, 2.72], $tol);
    /* quarter values are exact in binary → clean banker's at decimals=1 */
    $q = NumPower::array([0.25, 0.75, 1.25, 1.75], $dt);
    check("$dt round(,1) banker", NumPower::round($q, 1)->toArray(),
          [0.2, 0.8, 1.2, 1.8], $tol);
}

/* ── negative precision (round left of the point), banker's ── */
foreach (['float32' => 1e-3, 'float64' => 1e-9] as $dt => $tol) {
    $a = NumPower::array([1234.5678, 1250.0, 1350.0, -1250.0], $dt);
    check("$dt round(,-2)", NumPower::round($a, -2)->toArray(),
          [1200.0, 1200.0, 1400.0, -1200.0], $tol);
}

/* ── integer dtypes: identity for any precision, dtype preserved ── */
foreach (['int8','uint8','int16','uint16','int32','uint32','int64','uint64'] as $dt) {
    $a = NumPower::array([12, 25, 37], $dt);
    $r0 = NumPower::round($a, 0);
    check("$dt int identity(,0)", $r0->toArray(), [12, 25, 37]);
    check("$dt int dtype(,0)", dt_of($r0), $dt);
    $rn = NumPower::round($a, -1);
    check("$dt int identity(,-1)", $rn->toArray(), [12, 25, 37]);
    check("$dt int dtype(,-1)", dt_of($rn), $dt);
}

/* ── float4 / float8 (narrow floats compute via float32, cast back) ── */
foreach (['float4', 'float8'] as $dt) {
    $a = NumPower::array([0.0, 1.0, 2.0, 3.0], $dt);
    $r = NumPower::round($a, 0);
    check("$dt narrow dtype", dt_of($r), $dt);
    check("$dt narrow values", $r->toArray(), [0.0, 1.0, 2.0, 3.0], 0.1);
}

/* ── multi-dimensional shapes ── */
$m2 = NumPower::array([[0.5, 1.5], [2.5, 3.5]], 'float64');
check("2-D round", NumPower::round($m2, 0)->toArray(), [[0.0, 2.0], [2.0, 4.0]], 1e-12);
$m3 = NumPower::array([[[1.49, 1.51]], [[2.5, -2.5]]], 'float64');
check("3-D round", NumPower::round($m3, 0)->toArray(), [[[1.0, 2.0]], [[2.0, -2.0]]], 1e-12);
$m4 = NumPower::array([[[[0.25]]], [[[0.75]]]], 'float64');
check("4-D round(,1)", NumPower::round($m4, 1)->toArray(), [[[[0.2]]], [[[0.8]]]], 1e-9);

/* ── 0-D scalar inputs collapse to a PHP scalar ── */
check("0-D round(2.5)", NumPower::round(NumPower::array(2.5), 0), 2.0, 1e-12);
check("0-D round(2.5,float64)", NumPower::round(NumPower::array(2.5, 'float64'), 0), 2.0, 1e-12);
check("0-D round(3.14159,2)", NumPower::round(NumPower::array(3.14159, 'float64'), 2), 3.14, 1e-12);

/* ── empty arrays keep shape + dtype ── */
foreach (['float32', 'float64', 'int32'] as $dt) {
    $e = NumPower::zeros([0, 4], $dt);
    $r = NumPower::round($e, 2);
    check("$dt empty shape", $r->shape(), [0, 4]);
    check("$dt empty dtype", dt_of($r), $dt);
}

/* ── bare numeric-string intake (single-call fp128 / uint64) ── */
check("string fp128 round(,2)", (string)NumPower::round("3.14159", 2), "3.14", 1e-12);
check("string fp128 banker",    (string)NumPower::round("2.5", 0),     "2",    1e-12);
check("string uint64 identity", (string)NumPower::round("18446744073709551615", 0),
      "18446744073709551615");
check("string int64 identity",  (string)NumPower::round("-123", -1), "-123");

/* ── boundary values: NaN / Inf propagate; dtype extremes ── */
$edge = NumPower::array([INF, -INF, NAN, 0.0], 'float64');
$re = NumPower::round($edge, 0)->toArray();
check("round(+Inf)", is_infinite($re[0]) && $re[0] > 0, true);
check("round(-Inf)", is_infinite($re[1]) && $re[1] < 0, true);
check("round(NaN)",  is_nan($re[2]), true);
check("round(0.0)",  $re[3], 0.0, 1e-15);
$re2 = NumPower::round($edge, 2)->toArray();
check("round(NaN,2)", is_nan($re2[2]), true);
check("round(+Inf,2)", is_infinite($re2[0]) && $re2[0] > 0, true);

/* int dtype extremes survive the identity copy unchanged */
$ext = NumPower::array([127, -128], 'int8');
check("int8 extremes identity", NumPower::round($ext, 0)->toArray(), [127, -128]);
$uext = NumPower::array([255, 0], 'uint8');
check("uint8 extremes identity", NumPower::round($uext, -2)->toArray(), [255, 0]);

/* ── PyTorch parity on the classic scaling artifact ──────────────────────
   round follows PyTorch's `nearbyint(x·10^d)/10^d` kernel, NOT Python's
   decimal-correct built-in round(). 2.675 is stored just below 2.675, but
   2.675·100 rounds up to the exact tie 267.5, which banker's rounding sends
   to 268 → 2.68 (Python's round() would give 2.67). 1.005·100 stays below
   100.5 → 100 → 1.0. These results are IEEE-deterministic across platforms. */
check("parity round(2.675,2)=2.68", NumPower::round(NumPower::array([2.675], 'float64'), 2)->toArray(), [2.68], 1e-12);
check("parity round(1.005,2)=1.0",  NumPower::round(NumPower::array([1.005], 'float64'), 2)->toArray(), [1.0], 1e-12);

/* ── bare PHP float / int scalar intake (the float|int half of the union) ── */
check("raw float scalar", NumPower::round(2.5, 0), 2.0, 1e-6);
check("raw int scalar",   NumPower::round(25, 0),  25.0);
check("raw float default precision", NumPower::round(3.5), 4.0, 1e-6);

/* ── string-intake dtype-inference boundaries (single-call precision) ── */
check("string >UINT64_MAX -> fp128", (string)NumPower::round("18446744073709551616", 0),
      "18446744073709551616");
check("string >INT64_MAX -> uint64", (string)NumPower::round("9223372036854775808", 0),
      "9223372036854775808");
check("string +inf -> fp128", (string)NumPower::round("inf"), "inf");
check("string nan -> fp128",  (string)NumPower::round("nan"), "nan");

/* ── malformed / empty / whitespace strings throw ── */
foreach (['', ' ', 'abc', '1.5abc', '0x1F', '1,5'] as $bad) {
    try {
        NumPower::round($bad, 0);
        echo "FAIL round('$bad') did not throw\n";
    } catch (\Throwable $e) {
        echo "OK round('$bad') throws\n";
    }
}

/* ── negative precision on fp16 and fp128 (banker's), not just f32/f64 ── */
check("fp16 round(,-2) banker", NumPower::round(NumPower::array([150.0, 250.0], 'float16'), -2)->toArray(),
      [200.0, 200.0], 5e-2);
check("fp128 round(,-2) banker", NumPower::round(NumPower::array(['1234.5678', '1250.0'], 'float128'), -2)->toArray(),
      [1200.0, 1200.0], 1e-9);

/* ── fp128 ±Inf / NaN pass through round at non-zero precision (regression:
   the DD multiply used to corrupt ±Inf to NaN) ── */
check("fp128 round(inf,2) preserves Inf/NaN",
      NumPower::round(NumPower::array(['inf', '-inf', 'nan'], 'float128'), 2)->toArray(),
      ['inf', '-inf', 'nan']);

echo "DONE\n";
?>
--EXPECT--
OK float16 halves->even
OK float16 dtype preserved
OK float16 default precision
OK float32 halves->even
OK float32 dtype preserved
OK float32 default precision
OK float64 halves->even
OK float64 dtype preserved
OK float64 default precision
OK float128 halves->even
OK float128 dtype preserved
OK float128 default precision
OK float16 round(,2)
OK float16 round(,1) banker
OK float32 round(,2)
OK float32 round(,1) banker
OK float64 round(,2)
OK float64 round(,1) banker
OK float128 round(,2)
OK float128 round(,1) banker
OK float32 round(,-2)
OK float64 round(,-2)
OK int8 int identity(,0)
OK int8 int dtype(,0)
OK int8 int identity(,-1)
OK int8 int dtype(,-1)
OK uint8 int identity(,0)
OK uint8 int dtype(,0)
OK uint8 int identity(,-1)
OK uint8 int dtype(,-1)
OK int16 int identity(,0)
OK int16 int dtype(,0)
OK int16 int identity(,-1)
OK int16 int dtype(,-1)
OK uint16 int identity(,0)
OK uint16 int dtype(,0)
OK uint16 int identity(,-1)
OK uint16 int dtype(,-1)
OK int32 int identity(,0)
OK int32 int dtype(,0)
OK int32 int identity(,-1)
OK int32 int dtype(,-1)
OK uint32 int identity(,0)
OK uint32 int dtype(,0)
OK uint32 int identity(,-1)
OK uint32 int dtype(,-1)
OK int64 int identity(,0)
OK int64 int dtype(,0)
OK int64 int identity(,-1)
OK int64 int dtype(,-1)
OK uint64 int identity(,0)
OK uint64 int dtype(,0)
OK uint64 int identity(,-1)
OK uint64 int dtype(,-1)
OK float4 narrow dtype
OK float4 narrow values
OK float8 narrow dtype
OK float8 narrow values
OK 2-D round
OK 3-D round
OK 4-D round(,1)
OK 0-D round(2.5)
OK 0-D round(2.5,float64)
OK 0-D round(3.14159,2)
OK float32 empty shape
OK float32 empty dtype
OK float64 empty shape
OK float64 empty dtype
OK int32 empty shape
OK int32 empty dtype
OK string fp128 round(,2)
OK string fp128 banker
OK string uint64 identity
OK string int64 identity
OK round(+Inf)
OK round(-Inf)
OK round(NaN)
OK round(0.0)
OK round(NaN,2)
OK round(+Inf,2)
OK int8 extremes identity
OK uint8 extremes identity
OK parity round(2.675,2)=2.68
OK parity round(1.005,2)=1.0
OK raw float scalar
OK raw int scalar
OK raw float default precision
OK string >UINT64_MAX -> fp128
OK string >INT64_MAX -> uint64
OK string +inf -> fp128
OK string nan -> fp128
OK round('') throws
OK round(' ') throws
OK round('abc') throws
OK round('1.5abc') throws
OK round('0x1F') throws
OK round('1,5') throws
OK fp16 round(,-2) banker
OK fp128 round(,-2) banker
OK fp128 round(inf,2) preserves Inf/NaN
DONE
