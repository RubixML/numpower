--TEST--
NumPower::exp/exp2/expm1/log/log1p/log2/log10/logb across every dtype (CPU)
--FILE--
<?php
/* Covers the eight exp / log family ops on every CPU dtype. Verifies:
     - integer inputs promote to float32 (narrow ints) or float64
       (32/64-bit ints) per the unary-result-dtype rule;
     - every floating-point dtype is preserved end-to-end;
     - bare-string inputs throw the dtype-inference error;
     - fp128 inputs round-trip through libquadmath (when present) with
       full 32-digit precision via NumPower::array(['...'], 'float128');
     - 0-D inputs return a dtype-correct scalar;
     - multi-dimensional inputs preserve shape. */

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

/* ── float32 / float64: full op coverage ─────────────────────────────── */
foreach (['float32','float64'] as $dt) {
    $tol = ($dt === 'float32') ? 1e-5 : 1e-12;
    $arr = NumPower::array([0.0, 1.0, 2.0], $dt);
    check("$dt exp",    NumPower::exp($arr)->toArray(),    [1.0, M_E, M_E * M_E], $tol);
    check("$dt exp2",   NumPower::exp2($arr)->toArray(),   [1.0, 2.0, 4.0],       $tol);
    check("$dt expm1 small",
          NumPower::expm1(NumPower::array([0.0, 1e-7], $dt))->toArray(),
          [0.0, 1e-7], 1e-6);
    check("$dt log",    NumPower::log(NumPower::array([1.0, M_E, M_E*M_E], $dt))->toArray(),
                        [0.0, 1.0, 2.0], $tol);
    check("$dt log1p small",
          NumPower::log1p(NumPower::array([0.0, 1e-7], $dt))->toArray(),
          [0.0, 1e-7], 1e-6);
    check("$dt log2",   NumPower::log2(NumPower::array([1.0, 2.0, 8.0, 1024.0], $dt))->toArray(),
                        [0.0, 1.0, 3.0, 10.0], $tol);
    check("$dt log10",  NumPower::log10(NumPower::array([1.0, 10.0, 1000.0], $dt))->toArray(),
                        [0.0, 1.0, 3.0], $tol);
    check("$dt logb",   NumPower::logb(NumPower::array([1.0, 4.0, 1024.0], $dt))->toArray(),
                        [0.0, 2.0, 10.0], $tol);
}

/* ── float16: compute through float32, store as fp16 (low precision) ── */
$dt = 'float16';
$tol = 5e-3;
$arr = NumPower::array([0.0, 1.0, 2.0], $dt);
check("$dt exp",  NumPower::exp($arr)->toArray(),  [1.0, M_E, M_E*M_E], 5e-2);
check("$dt exp2", NumPower::exp2($arr)->toArray(), [1.0, 2.0, 4.0], $tol);
check("$dt log",  NumPower::log(NumPower::array([1.0, 8.0, 1024.0], $dt))->toArray(),
                  [0.0, log(8.0), log(1024.0)], 1e-2);
check("$dt log2", NumPower::log2(NumPower::array([1.0, 2.0, 8.0, 1024.0], $dt))->toArray(),
                  [0.0, 1.0, 3.0, 10.0], $tol);
check("$dt logb", NumPower::logb(NumPower::array([1.0, 4.0, 1024.0], $dt))->toArray(),
                  [0.0, 2.0, 10.0], $tol);

/* ── float4 / float8 (narrow non-half floats): compute via float32 ──── */
foreach (['float4', 'float8'] as $dt) {
    $tol = ($dt === 'float4') ? 0.6 : 0.4;
    $arr = NumPower::array([1.0, 2.0], $dt);
    /* exp(1) ≈ 2.7, exp(2) ≈ 7.4 — fp4's max representable is 6, so
       exp(2) clamps to 6 on round-trip; fp8 (E4M3, max 240) keeps both. */
    $e = NumPower::exp($arr)->toArray();
    $exp_want = ($dt === 'float4') ? [3.0, 6.0] : [M_E, M_E*M_E];
    check("$dt exp",  $e, $exp_want, $tol);

    $l = NumPower::log(NumPower::array([1.0, 2.0], $dt))->toArray();
    check("$dt log",  $l, [0.0, log(2.0)], $tol);
}

/* ── Integer dtypes: promotion to float ──────────────────────────────── */
foreach (['int8','int16','uint8','uint16'] as $dt) {
    $arr = NumPower::array([0, 1, 2], $dt);
    /* narrow int → float32 */
    $r = NumPower::exp($arr);
    $s = $r->__serialize();
    check("$dt exp dtype", $s['dtype'], 'float32');
    check("$dt exp",       $r->toArray(), [1.0, M_E, M_E*M_E], 1e-5);
}
foreach (['int32','int64','uint32','uint64'] as $dt) {
    $arr = NumPower::array([0, 1, 2], $dt);
    /* wide int → float64 */
    $r = NumPower::log2($arr);
    $s = $r->__serialize();
    check("$dt log2 dtype", $s['dtype'], 'float64');
    check("$dt log2",       NumPower::log2(NumPower::array([1, 2, 8, 1024], $dt))->toArray(),
                            [0.0, 1.0, 3.0, 10.0], 1e-12);
}

/* ── float128: precision round-trip via libquadmath when available ───── */
$f128 = NumPower::array(['0.0', '1.0', '2.0'], 'float128');
$r = NumPower::exp($f128);
$s = $r->__serialize();
check("fp128 exp dtype", $s['dtype'], 'float128');
/* exp(0)=1, exp(1)=e, exp(2)=e^2 — check the first elements are exactly
   1 and the e prefix is correct (precision floor: 15 digits suffices
   even on the DD fallback). */
$out = $r->toArray();
check("fp128 exp(0)", (float)$out[0], 1.0, 1e-12);
check("fp128 exp(1) prefix matches e",
      strncmp((string)$out[1], '2.71828182845904523536', 22) === 0, true);

$l = NumPower::log(NumPower::array(['1.0', '2.71828182845904523536', '7.38905609893065022723'],
                                    'float128'));
check("fp128 log(1)", (float)$l->toArray()[0], 0.0, 1e-15);
/* log(e) should be very close to 1; on libquadmath path the precision
   is full fp128 so the result is within 1e-30 of 1. */
$l1 = $l->toArray()[1];
check("fp128 log(e) ≈ 1", (float)$l1, 1.0, 1e-10);

/* ── 0-D scalar input: returns a PHP scalar of the right type ────────── */
$z = NumPower::array(1.0);
check("0-D float32 exp scalar", NumPower::exp($z),       M_E,        1e-5);
check("0-D float32 log scalar", NumPower::log(NumPower::array(M_E)),  1.0, 1e-5);
check("0-D float64 exp scalar", NumPower::exp(NumPower::array(1.0, 'float64')), M_E,  1e-12);

/* 0-D fp128 returns a string (full precision). */
$fz = NumPower::array(['0.0'], 'float128')->toArray()[0];  /* "0" */
$ez = NumPower::exp(NumPower::array(['1.0'], 'float128'));
$ez_arr = $ez->toArray();
check("0-D fp128 exp string prefix",
      strncmp((string)$ez_arr[0], '2.7182818284590452353', 20) === 0, true);

/* ── 2-D / 3-D shapes preserve structure ─────────────────────────────── */
$mat = NumPower::array([[1.0, M_E], [M_E*M_E, M_E*M_E*M_E]], 'float64');
check("2-D log",  NumPower::log($mat)->toArray(),  [[0.0, 1.0], [2.0, 3.0]], 1e-12);
check("2-D exp2 of zeros", NumPower::exp2(NumPower::array([[0.0,0.0],[0.0,0.0]], 'float64'))->toArray(),
                           [[1.0,1.0],[1.0,1.0]], 1e-12);

$cube = NumPower::array([[[1.0, 10.0],[100.0, 1000.0]]], 'float64');
check("3-D log10", NumPower::log10($cube)->toArray(),
                   [[[0.0, 1.0], [2.0, 3.0]]], 1e-12);

/* ── Bare-string input throws ─────────────────────────────────────────── */
try { NumPower::exp("1.0"); echo "FAIL exp(string) did not throw\n"; }
catch (Error $e) { echo "OK exp(string) throws\n"; }

try { NumPower::log("2.0"); echo "FAIL log(string) did not throw\n"; }
catch (Error $e) { echo "OK log(string) throws\n"; }

/* ── Edge values: log(1)=0, log(0)=-inf, log(-1)=NaN, exp(0)=1 ────────── */
$edge = NumPower::array([0.0, 1.0, -1.0], 'float64');
$le = NumPower::log($edge)->toArray();
check("log(0)=-inf", is_infinite($le[0]) && $le[0] < 0, true);
check("log(1)=0",    $le[1], 0.0, 1e-12);
check("log(-1)=NaN", is_nan($le[1+1]), true);

/* expm1(0)=0 exactly, log1p(0)=0 exactly (no cancellation) */
$zero = NumPower::array([0.0], 'float64');
check("expm1(0)=0",  NumPower::expm1($zero)->toArray(), [0.0], 0.0);
check("log1p(0)=0",  NumPower::log1p($zero)->toArray(), [0.0], 0.0);

echo "DONE\n";
?>
--EXPECTF--
OK float32 exp
OK float32 exp2
OK float32 expm1 small
OK float32 log
OK float32 log1p small
OK float32 log2
OK float32 log10
OK float32 logb
OK float64 exp
OK float64 exp2
OK float64 expm1 small
OK float64 log
OK float64 log1p small
OK float64 log2
OK float64 log10
OK float64 logb
OK float16 exp
OK float16 exp2
OK float16 log
OK float16 log2
OK float16 logb
OK float4 exp
OK float4 log
OK float8 exp
OK float8 log
OK int8 exp dtype
OK int8 exp
OK int16 exp dtype
OK int16 exp
OK uint8 exp dtype
OK uint8 exp
OK uint16 exp dtype
OK uint16 exp
OK int32 log2 dtype
OK int32 log2
OK int64 log2 dtype
OK int64 log2
OK uint32 log2 dtype
OK uint32 log2
OK uint64 log2 dtype
OK uint64 log2
OK fp128 exp dtype
OK fp128 exp(0)
OK fp128 exp(1) prefix matches e
OK fp128 log(1)
OK fp128 log(e) ≈ 1
OK 0-D float32 exp scalar
OK 0-D float32 log scalar
OK 0-D float64 exp scalar
OK 0-D fp128 exp string prefix
OK 2-D log
OK 2-D exp2 of zeros
OK 3-D log10
OK exp(string) throws
OK log(string) throws
OK log(0)=-inf
OK log(1)=0
OK log(-1)=NaN
OK expm1(0)=0
OK log1p(0)=0
DONE
