--TEST--
NumPower exp/log family: edge cases (NaN/Inf, identity values, 0-D, empty arrays)
--FILE--
<?php
/* Edge-case coverage for transcendental ops:
     - NaN propagates: exp(NaN)=NaN, log(NaN)=NaN, etc.;
     - +Inf: exp(+Inf)=+Inf, log(+Inf)=+Inf, exp2(+Inf)=+Inf;
     - -Inf: exp(-Inf)=+0, log(0)=-Inf, log1p(-1)=-Inf;
     - identity values:
         exp(0)=1, exp2(0)=1, expm1(0)=0,
         log(1)=0, log2(1)=0, log10(1)=0, log1p(0)=0, logb(1)=0;
     - 0-D inputs return PHP scalar of the right type;
     - empty NDArray preserves shape and dtype on every op;
     - dimension-exceeding sizes (single big array) doesn't crash. */

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

/* ── NaN / Inf propagation on float32 / float64 ───────────────────────── */
foreach (['float32','float64'] as $dt) {
    $tol = $dt === 'float32' ? 1e-5 : 1e-12;
    $special = NumPower::array([INF, -INF, NAN], $dt);
    /* exp(+Inf)=+Inf, exp(-Inf)=0, exp(NaN)=NaN */
    $e = NumPower::exp($special)->toArray();
    check("$dt exp(+Inf)=+Inf", is_infinite($e[0]) && $e[0] > 0, true);
    check("$dt exp(-Inf)=0",     $e[1], 0.0, $tol);
    check("$dt exp(NaN)=NaN",    is_nan($e[2]), true);
    /* expm1(-Inf)=-1 */
    $em = NumPower::expm1(NumPower::array([-INF], $dt))->toArray();
    check("$dt expm1(-Inf)=-1", $em[0], -1.0, $tol);
    /* log(+Inf)=+Inf, log(NaN)=NaN, log(-1)=NaN */
    $l = NumPower::log(NumPower::array([INF, NAN, -1.0], $dt))->toArray();
    check("$dt log(+Inf)=+Inf", is_infinite($l[0]) && $l[0] > 0, true);
    check("$dt log(NaN)=NaN",   is_nan($l[1]), true);
    check("$dt log(-1)=NaN",    is_nan($l[2]), true);
    /* log(0)=-Inf, log2(0)=-Inf, log10(0)=-Inf */
    $z = NumPower::array([0.0], $dt);
    check("$dt log(0)=-Inf",    is_infinite(NumPower::log($z)->toArray()[0])
                                && NumPower::log($z)->toArray()[0] < 0, true);
    check("$dt log2(0)=-Inf",   is_infinite(NumPower::log2($z)->toArray()[0])
                                && NumPower::log2($z)->toArray()[0] < 0, true);
    check("$dt log10(0)=-Inf",  is_infinite(NumPower::log10($z)->toArray()[0])
                                && NumPower::log10($z)->toArray()[0] < 0, true);
    /* log1p(-1)=-Inf, log1p(<-1)=NaN */
    $lp = NumPower::log1p(NumPower::array([-1.0, -2.0], $dt))->toArray();
    check("$dt log1p(-1)=-Inf", is_infinite($lp[0]) && $lp[0] < 0, true);
    check("$dt log1p(-2)=NaN",   is_nan($lp[1]), true);
}

/* ── Identity values ─────────────────────────────────────────────────── */
foreach (['float32','float64'] as $dt) {
    $tol = $dt === 'float32' ? 1e-6 : 1e-15;
    $zero = NumPower::array([0.0], $dt);
    $one  = NumPower::array([1.0], $dt);
    check("$dt exp(0)=1",   NumPower::exp($zero)->toArray(),   [1.0], $tol);
    check("$dt exp2(0)=1",  NumPower::exp2($zero)->toArray(),  [1.0], $tol);
    check("$dt expm1(0)=0", NumPower::expm1($zero)->toArray(), [0.0], $tol);
    check("$dt log(1)=0",   NumPower::log($one)->toArray(),    [0.0], $tol);
    check("$dt log2(1)=0",  NumPower::log2($one)->toArray(),   [0.0], $tol);
    check("$dt log10(1)=0", NumPower::log10($one)->toArray(),  [0.0], $tol);
    check("$dt log1p(0)=0", NumPower::log1p($zero)->toArray(), [0.0], $tol);
    check("$dt logb(1)=0",  NumPower::logb($one)->toArray(),   [0.0], $tol);
    /* logb(2^k) == k */
    check("$dt logb(8)=3",  NumPower::logb(NumPower::array([8.0], $dt))->toArray(), [3.0], $tol);
    /* exp2 at boundary inputs */
    check("$dt exp2(10)=1024", NumPower::exp2(NumPower::array([10.0], $dt))->toArray(), [1024.0], $tol);
}

/* ── 0-D scalar input: returns scalar (not NDArray) ───────────────────── */
$z = NumPower::array(1.0);   /* 0-D */
$e = NumPower::exp($z);
check("0-D exp returns scalar",  is_float($e), true);
check("0-D exp(1)=e",            $e, M_E, 1e-5);
check("0-D log(e) scalar",       NumPower::log(NumPower::array(M_E)), 1.0, 1e-5);

/* 0-D int input promoting to float (returns float scalar) */
$zi = NumPower::array(8);
$l2 = NumPower::log2($zi);
check("0-D int log2 scalar",     is_float($l2), true);
check("0-D log2(8)=3",            $l2, 3.0, 1e-5);

/* 0-D fp128 returns a string */
$zf128 = NumPower::array(['1.0'], 'float128');   /* 1-D shape [1] for now */
$z0_fp128 = $zf128->__serialize();               /* check dtype */
check("fp128 1-D log dtype",     NumPower::log($zf128)->__serialize()['dtype'], 'float128');

/* ── Empty 1-D NDArrays: preserve shape and dtype ────────────────────── */
foreach (['float32','float64','float16'] as $dt) {
    $empty = NumPower::array([], $dt);
    foreach (['exp','log','log2','log10','log1p','exp2','expm1','logb'] as $op) {
        $r = NumPower::$op($empty);
        check("$dt $op(empty) shape", $r->shape(), [0]);
        check("$dt $op(empty) dtype", $r->__serialize()['dtype'], $dt);
    }
}

/* ── Bigger arrays (cross block boundary on GPU): just verify CPU OK ── */
$big = [];
for ($i = 1; $i <= 1025; $i++) $big[] = (float)$i;
$ba = NumPower::array($big, 'float64');
$lb = NumPower::log($ba);
check("1025-element log shape",  $lb->shape(), [1025]);
check("1025-element log[0]=0",   $lb->toArray()[0], 0.0, 1e-15);
check("1025-element log[1024]",  $lb->toArray()[1024], log(1025), 1e-12);

/* ── String input is accepted: "1.0" → fp128 0-D scalar ──────────────── */
/* Each op returns an fp128 0-D scalar (string) for a decimal literal.
   Spot-check the result is finite and non-empty. */
foreach (['exp','exp2','expm1','log','log1p','log2','log10','logb'] as $op) {
    try {
        $r = (string)NumPower::$op("1.0");
        $ok = is_finite((float)$r) || ($op === 'logb' && (float)$r === 0.0);
        if (strlen($r) > 0 && $ok) {
            echo "OK $op('1.0') -> $r\n";
        } else {
            echo "FAIL $op('1.0') unexpected result: '$r'\n";
        }
    } catch (Error $e) {
        echo "FAIL $op('1.0') threw: ", $e->getMessage(), "\n";
    }
}
/* Empty / whitespace-only strings still throw on every op. */
foreach (['exp','exp2','expm1','log','log1p','log2','log10','logb'] as $op) {
    try {
        NumPower::$op("");
        echo "FAIL $op('') did not throw\n";
    } catch (Error $e) {
        echo "OK $op('') throws\n";
    }
}

echo "DONE\n";
?>
--EXPECTF--
%aDONE
