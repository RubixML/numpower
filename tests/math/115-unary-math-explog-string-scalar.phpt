--TEST--
MATHEMATICAL FUNCTIONS + EXPONENTIAL & LOGARITHMIC: numeric-string scalars infer dtype loss-free (float128 / int64 / uint64)
--FILE--
<?php
/* Verifies the unary dispatcher (and clip) accept bare numeric strings
   on every op in:
     - MATHEMATICAL FUNCTIONS: abs, negative, positive, reciprocal,
       sign, sqrt, rsqrt, square, clip, sinc;
     - EXPONENTIAL & LOGARITHMIC: exp, exp2, expm1, log, log1p, log2,
       log10, logb.

   The dtype is inferred from the literal:
     - decimal point / `e`/`E` exponent / `inf` / `nan` → float128 (the
       only dtype that holds the full decimal precision the caller
       chose to express as a string);
     - non-negative integer literal whose magnitude exceeds INT64_MAX
       (i.e. > 19 digits, or 19 digits exceeding the INT64_MAX prefix)
       → uint64 (the only dtype that holds the full 64-bit range);
     - every other integer literal → int64.

   Empty / whitespace-only strings throw a clean error. Surrounding
   whitespace (mirrors strtod) is tolerated.
*/

function check($label, $got, $want, $tol = 0.0) {
    if (is_string($want) && is_string($got)) {
        if ($got === $want) { echo "OK $label\n"; return; }
        echo "FAIL $label: got=\"$got\" want=\"$want\"\n"; return;
    }
    if (is_bool($want)) {
        if ($got === $want) { echo "OK $label\n"; return; }
        echo "FAIL $label: got=", var_export($got, true),
             " want=", var_export($want, true), "\n"; return;
    }
    $gf = (float)$got; $wf = (float)$want;
    if (is_nan($gf) && is_nan($wf)) { echo "OK $label\n"; return; }
    if (is_infinite($gf) && is_infinite($wf) && (($gf<0)===($wf<0))) {
        echo "OK $label\n"; return;
    }
    if (abs($gf - $wf) <= $tol) { echo "OK $label\n"; return; }
    echo "FAIL $label: got=$gf want=$wf tol=$tol\n";
}

function expect_throw($label, $thunk) {
    try { $thunk(); echo "FAIL $label: no throw\n"; }
    catch (\Error $e) { echo "OK $label throws\n"; }
}

/* libquadmath gives full ~34-digit fp128 precision; DD-emulation
   collapses transcendentals to fp64 precision (~15 sig figs). Probe
   once and use a backend-appropriate tolerance everywhere below. */
function has_libquadmath() {
    $probe = (string)(new NDArray(['9.999999999999999999999999999999999e-10'],
                                  'float128'));
    return strpos($probe, '9.9999999999999999999999999999999') !== false;
}
$LQ = has_libquadmath();
$FP128_TOL_DECIMAL = $LQ ? 1e-30 : 1e-10;   /* arithmetic on fp128 */
$FP128_TOL_TRANSC  = $LQ ? 1e-30 : 1e-10;   /* transcendentals on fp128 */

/* ── Dtype-inference probes via NumPower::sign ──────────────────────────
   sign() preserves the input dtype and returns the correct dtype-aware
   scalar: string for fp128/uint64, int for the remaining ints, float
   for the remaining floats. Drives the dtype detection. */
check("sign('1.5') is string (fp128)",       is_string(NumPower::sign('1.5')),  true);
check("sign('-1.5') is string (fp128)",      is_string(NumPower::sign('-1.5')), true);
check("sign('1e10') is string (fp128)",      is_string(NumPower::sign('1e10')), true);
check("sign('inf') is string (fp128)",       is_string(NumPower::sign('inf')),  true);
check("sign('5') is int (int64)",            is_int(NumPower::sign('5')),       true);
check("sign('-5') is int (int64)",           is_int(NumPower::sign('-5')),      true);
check("sign('0') is int (int64)",            is_int(NumPower::sign('0')),       true);
/* INT64_MAX = 9223372036854775807, INT64_MAX+1 → uint64 */
check("sign('9223372036854775807') int64",
      is_int(NumPower::sign('9223372036854775807')),   true);
check("sign('9223372036854775808') uint64",
      is_string(NumPower::sign('9223372036854775808')),true);
check("sign('18446744073709551615') uint64",
      is_string(NumPower::sign('18446744073709551615')),true);
/* Negative magnitude can never fit uint64 → must stay int64 (signed). */
check("sign('-18446744073709551615') int64",
      is_int(NumPower::sign('-18446744073709551615')), true);

/* ── MATHEMATICAL FUNCTIONS ───────────────────────────────────────────── */

/* abs: dtype-preserving on every input. */
check("abs('-3.5') (fp128)",                 (float)NumPower::abs('-3.5'),  3.5, $FP128_TOL_DECIMAL);
check("abs('-100') (int64)",                 NumPower::abs('-100'),         100);
check("abs('18446744073709551615') (uint64)",NumPower::abs('18446744073709551615'),
                                              '18446744073709551615');

/* negative: dtype-preserving; unsigned wraps modulo 2^N. */
check("negative('3.5') (fp128)",             (float)NumPower::negative('3.5'), -3.5, $FP128_TOL_DECIMAL);
check("negative('100') (int64)",             NumPower::negative('100'),        -100);
/* negate(UINT64_MAX) wraps to 1 (numpy semantics on unsigned ints). */
check("negative('18446744073709551615') (uint64) wraps to 1",
      NumPower::negative('18446744073709551615'), '1');

/* positive: identity, preserves dtype. */
check("positive('3.5') (fp128)",             (float)NumPower::positive('3.5'),  3.5, $FP128_TOL_DECIMAL);
check("positive('-100') (int64)",            NumPower::positive('-100'),        -100);
check("positive('UINT64_MAX-1') (uint64)",   NumPower::positive('18446744073709551614'),
                                              '18446744073709551614');

/* reciprocal: integer inputs widen to float (narrow → float32, wider →
   float64). fp128 input stays fp128. */
check("reciprocal('2.0') (fp128)",           (float)NumPower::reciprocal('2.0'), 0.5, $FP128_TOL_DECIMAL);
check("reciprocal('4') widens to float64",   (float)NumPower::reciprocal('4'),   0.25, 1e-12);
check("reciprocal('18446744073709551615') widens to float64",
      (float)NumPower::reciprocal('18446744073709551615'), 1.0/1.8446744073709552e19, 1e-25);

/* sign: -1 / 0 / 1, preserves dtype. */
check("sign('-7.5')",                        NumPower::sign('-7.5'),  '-1');
check("sign('0.0')",                         NumPower::sign('0.0'),   '0');
check("sign('7.5')",                         NumPower::sign('7.5'),   '1');
check("sign('0') (int64)",                   NumPower::sign('0'),     0);
check("sign('-1') (int64)",                  NumPower::sign('-1'),    -1);
check("sign('18446744073709551615') (uint64)",
      NumPower::sign('18446744073709551615'), '1');

/* sqrt: integer inputs widen to float; fp128 stays fp128. */
check("sqrt('4.0') (fp128)",                 (float)NumPower::sqrt('4.0'),  2.0, $FP128_TOL_DECIMAL);
check("sqrt('16') widens to float",          (float)NumPower::sqrt('16'),   4.0, 1e-12);
check("sqrt('18446744073709551615') widens to float64",
      (float)NumPower::sqrt('18446744073709551615'), 4.294967296e9, 1.0);

/* rsqrt: same widening as sqrt. */
check("rsqrt('4.0') (fp128)",                (float)NumPower::rsqrt('4.0'), 0.5, $FP128_TOL_DECIMAL);
check("rsqrt('4') widens to float",          (float)NumPower::rsqrt('4'),   0.5, 1e-12);

/* square: preserves dtype; integer dtypes wrap modulo 2^N. */
check("square('3.0') (fp128)",               (float)NumPower::square('3.0'), 9.0, $FP128_TOL_DECIMAL);
check("square('10') (int64)",                NumPower::square('10'),         100);
/* (UINT64_MAX)^2 wraps to 1 (since UINT64_MAX = 2^64-1, mod 2^64 → 1). */
check("square('18446744073709551615') (uint64) wraps to 1",
      NumPower::square('18446744073709551615'), '1');

/* clip: bare string array with int/float/string bounds. */
check("clip('5.5', 0, 1) (fp128)",          (float)NumPower::clip('5.5', 0, 1), 1.0, $FP128_TOL_DECIMAL);
check("clip('-100', -10, 10) (int64)",       NumPower::clip('-100', -10, 10),  -10);
check("clip('18446744073709551615','0','1') (uint64) clamps to 1",
      NumPower::clip('18446744073709551615', '0', '1'), '1');
check("clip('1.234567890123456789012345e30','0','1e29') (fp128)",
      (float)NumPower::clip('1.234567890123456789012345e30', '0', '1e29')/1e29, 1.0, 1e-10);

/* sinc: integer widens to float; fp128 stays fp128. */
check("sinc('0.0') (fp128)=1",               (float)NumPower::sinc('0.0'),  1.0, $FP128_TOL_DECIMAL);
check("sinc('1') widens to float (≈0)",      (float)NumPower::sinc('1'),    0.0, 1e-12);

/* ── EXPONENTIAL & LOGARITHMIC ────────────────────────────────────────── */

/* exp: fp128 stays fp128, integer widens. e ≈ 2.71828... */
check("exp('1.0') (fp128) ≈ e",              (float)NumPower::exp('1.0'),   M_E,  $FP128_TOL_TRANSC);
check("exp('0') widens to float (1.0)",      (float)NumPower::exp('0'),     1.0, 1e-12);
check("exp('18446744073709551615') widens; result = +inf",
      is_infinite((float)NumPower::exp('18446744073709551615')), true);

/* exp2: 2^x. */
check("exp2('4.0') (fp128) == 16",           (float)NumPower::exp2('4.0'), 16.0, $FP128_TOL_TRANSC);
check("exp2('10') widens to float (1024)",   (float)NumPower::exp2('10'), 1024.0, 1e-12);

/* expm1: precise near zero. */
check("expm1('0.0') (fp128) == 0",           (float)NumPower::expm1('0.0'),  0.0, $FP128_TOL_TRANSC);
check("expm1('1') widens (≈ e-1)",           (float)NumPower::expm1('1'),    M_E - 1.0, 1e-12);

/* log: ln. */
check("log('2.718281828459045235360287') (fp128) ≈ 1",
      (float)NumPower::log('2.718281828459045235360287'), 1.0, $FP128_TOL_TRANSC);
check("log('1') widens to float (0)",        (float)NumPower::log('1'),       0.0, 1e-12);
/* PHP's (float)"-inf" yields 0, so compare the fp128 string form instead. */
check("log('0.0') (fp128) == -inf",
      stripos((string)NumPower::log('0.0'), 'inf') !== false, true);

/* log1p: precise near zero. */
check("log1p('0.0') (fp128) == 0",           (float)NumPower::log1p('0.0'),  0.0, $FP128_TOL_TRANSC);
check("log1p('1') widens (ln(2))",           (float)NumPower::log1p('1'),    log(2.0), 1e-12);

/* log2. */
check("log2('1024.0') (fp128) == 10",        (float)NumPower::log2('1024.0'), 10.0, $FP128_TOL_TRANSC);
check("log2('1024') widens to float (10)",   (float)NumPower::log2('1024'),   10.0, 1e-12);
check("log2('18446744073709551615') widens (≈ 64)",
      (float)NumPower::log2('18446744073709551615'), 64.0, 1e-6);

/* log10. */
check("log10('1000.0') (fp128) == 3",        (float)NumPower::log10('1000.0'), 3.0, $FP128_TOL_TRANSC);
check("log10('1000') widens to float",       (float)NumPower::log10('1000'),   3.0, 1e-12);

/* logb. */
check("logb('1024.0') (fp128) == 10",        (float)NumPower::logb('1024.0'), 10.0, $FP128_TOL_TRANSC);
check("logb('1024') widens to float",        (float)NumPower::logb('1024'),    10.0, 1e-12);

/* ── Surrounding whitespace tolerated (mirrors strtod) ───────────────── */
check("abs(' 1.5 ') (fp128) ≈ 1.5",          (float)NumPower::abs(' 1.5 '), 1.5, $FP128_TOL_DECIMAL);
check('abs(tab+nl wrap) (int64)',            NumPower::abs("\t-100\n"),     100);

/* ── Leading '+' tolerated on integer literals. ──────────────────────── */
check("abs('+100') (int64)",                 NumPower::abs('+100'),         100);
check("abs('+18446744073709551615') (uint64)",
      NumPower::abs('+18446744073709551615'), '18446744073709551615');

/* ── Empty / whitespace-only strings throw on every op ───────────────── */
foreach (['abs','negative','positive','reciprocal','sign','sqrt','rsqrt',
          'square','sinc',
          'exp','exp2','expm1','log','log1p','log2','log10','logb'] as $op) {
    expect_throw("$op('')",    fn() => NumPower::$op(''));
    expect_throw("$op('   ')", fn() => NumPower::$op('   '));
}
expect_throw("clip('',0,1)",   fn() => NumPower::clip('', 0, 1));
expect_throw("clip('  ',0,1)", fn() => NumPower::clip('  ', 0, 1));

echo "DONE\n";
?>
--EXPECT--
OK sign('1.5') is string (fp128)
OK sign('-1.5') is string (fp128)
OK sign('1e10') is string (fp128)
OK sign('inf') is string (fp128)
OK sign('5') is int (int64)
OK sign('-5') is int (int64)
OK sign('0') is int (int64)
OK sign('9223372036854775807') int64
OK sign('9223372036854775808') uint64
OK sign('18446744073709551615') uint64
OK sign('-18446744073709551615') int64
OK abs('-3.5') (fp128)
OK abs('-100') (int64)
OK abs('18446744073709551615') (uint64)
OK negative('3.5') (fp128)
OK negative('100') (int64)
OK negative('18446744073709551615') (uint64) wraps to 1
OK positive('3.5') (fp128)
OK positive('-100') (int64)
OK positive('UINT64_MAX-1') (uint64)
OK reciprocal('2.0') (fp128)
OK reciprocal('4') widens to float64
OK reciprocal('18446744073709551615') widens to float64
OK sign('-7.5')
OK sign('0.0')
OK sign('7.5')
OK sign('0') (int64)
OK sign('-1') (int64)
OK sign('18446744073709551615') (uint64)
OK sqrt('4.0') (fp128)
OK sqrt('16') widens to float
OK sqrt('18446744073709551615') widens to float64
OK rsqrt('4.0') (fp128)
OK rsqrt('4') widens to float
OK square('3.0') (fp128)
OK square('10') (int64)
OK square('18446744073709551615') (uint64) wraps to 1
OK clip('5.5', 0, 1) (fp128)
OK clip('-100', -10, 10) (int64)
OK clip('18446744073709551615','0','1') (uint64) clamps to 1
OK clip('1.234567890123456789012345e30','0','1e29') (fp128)
OK sinc('0.0') (fp128)=1
OK sinc('1') widens to float (≈0)
OK exp('1.0') (fp128) ≈ e
OK exp('0') widens to float (1.0)
OK exp('18446744073709551615') widens; result = +inf
OK exp2('4.0') (fp128) == 16
OK exp2('10') widens to float (1024)
OK expm1('0.0') (fp128) == 0
OK expm1('1') widens (≈ e-1)
OK log('2.718281828459045235360287') (fp128) ≈ 1
OK log('1') widens to float (0)
OK log('0.0') (fp128) == -inf
OK log1p('0.0') (fp128) == 0
OK log1p('1') widens (ln(2))
OK log2('1024.0') (fp128) == 10
OK log2('1024') widens to float (10)
OK log2('18446744073709551615') widens (≈ 64)
OK log10('1000.0') (fp128) == 3
OK log10('1000') widens to float
OK logb('1024.0') (fp128) == 10
OK logb('1024') widens to float
OK abs(' 1.5 ') (fp128) ≈ 1.5
OK abs(tab+nl wrap) (int64)
OK abs('+100') (int64)
OK abs('+18446744073709551615') (uint64)
OK abs('') throws
OK abs('   ') throws
OK negative('') throws
OK negative('   ') throws
OK positive('') throws
OK positive('   ') throws
OK reciprocal('') throws
OK reciprocal('   ') throws
OK sign('') throws
OK sign('   ') throws
OK sqrt('') throws
OK sqrt('   ') throws
OK rsqrt('') throws
OK rsqrt('   ') throws
OK square('') throws
OK square('   ') throws
OK sinc('') throws
OK sinc('   ') throws
OK exp('') throws
OK exp('   ') throws
OK exp2('') throws
OK exp2('   ') throws
OK expm1('') throws
OK expm1('   ') throws
OK log('') throws
OK log('   ') throws
OK log1p('') throws
OK log1p('   ') throws
OK log2('') throws
OK log2('   ') throws
OK log10('') throws
OK log10('   ') throws
OK logb('') throws
OK logb('   ') throws
OK clip('',0,1) throws
OK clip('  ',0,1) throws
DONE
