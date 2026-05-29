--TEST--
Unary string-scalar intake: strict syntactic validation + overflow escalation
--FILE--
<?php
/* The string-scalar inference helper (ndarray_infer_dtype_from_string)
   was hardened to reject any literal that is not a syntactically
   complete numeric value, and to escalate non-negative integer
   literals above UINT64_MAX to fp128 instead of silently saturating
   at 2^64 - 1.

   Failure modes covered:
     - empty string                  → "got an empty value"
     - whitespace-only string        → "got a whitespace-only value"
     - hexadecimal literal ("0xff")  → "got malformed literal"
     - random characters             → malformed
     - partial parse ("1.5.5")       → malformed
     - trailing junk  ("1.5a")       → malformed
     - foreign separators ("1,5")    → malformed
     - exponent without digits ("1.5e") → malformed
     - lone sign / dot ("+", "-", ".")    → malformed
     - sign with embedded space ("  -  1.5") → malformed
     - illegal chars in fp / exp positions → malformed

   Success modes covered:
     - integer above UINT64_MAX      → fp128 (no saturation)
     - integer up to UINT64_MAX      → uint64 (preserved exactly)
     - integer up to INT64_MAX       → int64
     - negative integer down to       → int64 (signed two's complement)
       INT64_MIN
     - case-insensitive inf / nan / infinity literals → fp128
*/

function expect_throw($label, $thunk, $expected_fragment = null) {
    try {
        $thunk();
        echo "FAIL $label: no throw\n";
    } catch (\Error $e) {
        if ($expected_fragment === null ||
            strpos($e->getMessage(), $expected_fragment) !== false) {
            echo "OK $label throws\n";
        } else {
            echo "FAIL $label: wrong message: ", $e->getMessage(), "\n";
        }
    }
}

/* ── Empty / whitespace classification ──────────────────────────────── */
expect_throw("abs('')",      fn() => NumPower::abs(''),      'empty value');
expect_throw("abs('   ')",   fn() => NumPower::abs('   '),   'whitespace-only');
expect_throw("abs(\"\\t\\n\")", fn() => NumPower::abs("\t\n"), 'whitespace-only');
expect_throw("clip('',0,1)", fn() => NumPower::clip('', 0, 1), 'empty value');

/* ── Hexadecimal / random / mid-string non-numeric chars ────────────── */
expect_throw("abs('0xff')", fn() => NumPower::abs('0xff'), 'malformed literal');
expect_throw("abs('abc')",  fn() => NumPower::abs('abc'),  'malformed literal');
expect_throw("abs('1.5.5')",fn() => NumPower::abs('1.5.5'),'malformed literal');
expect_throw("abs('1.5a')", fn() => NumPower::abs('1.5a'), 'malformed literal');
expect_throw("abs('1,5')",  fn() => NumPower::abs('1,5'),  'malformed literal');
expect_throw("abs('1.5e')", fn() => NumPower::abs('1.5e'), 'malformed literal');

/* ── Lone sign / dot ────────────────────────────────────────────────── */
expect_throw("abs('+')",  fn() => NumPower::abs('+'),  'malformed literal');
expect_throw("abs('-')",  fn() => NumPower::abs('-'),  'malformed literal');
expect_throw("abs('.')",  fn() => NumPower::abs('.'),  'malformed literal');

/* ── Sign + whitespace + digits (invalid form) ──────────────────────── */
expect_throw("abs('  -  1.5')",  fn() => NumPower::abs('  -  1.5'), 'malformed literal');
expect_throw("abs('-.')",        fn() => NumPower::abs('-.'),       'malformed literal');
expect_throw("abs('1.5e+')",     fn() => NumPower::abs('1.5e+'),    'malformed literal');
expect_throw("abs('1.5E-')",     fn() => NumPower::abs('1.5E-'),    'malformed literal');

/* ── Trailing junk after a complete number ──────────────────────────── */
expect_throw("abs('100xyz')",    fn() => NumPower::abs('100xyz'),   'malformed literal');
expect_throw("abs('inf!')",      fn() => NumPower::abs('inf!'),     'malformed literal');
expect_throw("abs('nanjunk')",   fn() => NumPower::abs('nanjunk'),  'malformed literal');

/* ── Inf / nan tokens (case-insensitive) ────────────────────────────── */
echo "abs('inf')=",      (string)NumPower::abs('inf'),      "\n";
echo "abs('Inf')=",      (string)NumPower::abs('Inf'),      "\n";
echo "abs('INF')=",      (string)NumPower::abs('INF'),      "\n";
echo "abs('infinity')=", (string)NumPower::abs('infinity'), "\n";
echo "abs('Infinity')=", (string)NumPower::abs('Infinity'), "\n";
echo "abs('-inf')=",     (string)NumPower::abs('-inf'),     "\n";
echo "abs('nan')[lower3]=", strtolower(substr((string)NumPower::abs('nan'),0,3)), "\n";
echo "abs('NaN')[lower3]=", strtolower(substr((string)NumPower::abs('NaN'),0,3)), "\n";

/* ── Magnitude escalation: non-negative integer > UINT64_MAX → fp128 ── */
/* Non-power-of-two huge magnitudes display differently per fp128 backend
   (libquadmath = full digits; double-double = rounded / scientific), so
   assert escalation-to-wide-string + magnitude portably for those; the
   power-of-two and round-decimal values below display identically on both. */
$r = (string)NumPower::abs('99999999999999999999');   /* 20 nines, not 2^k */
echo "abs('99999999999999999999') → fp128 ~1e20: ",
     (is_string($r) && abs((float)$r / 1e20 - 1.0) < 1e-15) ? "OK" : "FAIL($r)", "\n";
echo "abs('18446744073709551616')=", (string)NumPower::abs('18446744073709551616'),  "\n";   /* UINT64_MAX+1 = 2^64 */
echo "abs('100000000000000000000')=",(string)NumPower::abs('100000000000000000000'), "\n";   /* 10^20 */
$r30 = (string)NumPower::abs('1' . str_repeat('0', 30));   /* 10^30 */
echo "abs('1e30') → fp128 ~1e30: ",
     (is_string($r30) && abs((float)$r30 / 1e30 - 1.0) < 1e-15) ? "OK" : "FAIL($r30)", "\n";

/* ── Boundary: INT64_MAX / INT64_MAX+1 / UINT64_MAX / negative limits ── */
echo "abs('9223372036854775807')=", (string)NumPower::abs('9223372036854775807'), "\n";   /* INT64_MAX (int64 return) */
echo "abs('9223372036854775808')=", (string)NumPower::abs('9223372036854775808'), "\n";   /* INT64_MAX+1 (uint64) */
echo "abs('18446744073709551615')=",(string)NumPower::abs('18446744073709551615'),"\n";   /* UINT64_MAX (uint64) */
/* Negative magnitudes can't fit uint64 — must go int64 (signed). */
echo "abs('-9223372036854775808')=",(string)NumPower::abs('-9223372036854775808'),"\n";   /* INT64_MIN: wraps to itself */

/* ── Leading + sign / leading zeros ─────────────────────────────────── */
echo "abs('+100')=",                  (string)NumPower::abs('+100'),                   "\n";
echo "abs('00000000000000000100')=",  (string)NumPower::abs('00000000000000000100'),   "\n";
echo "abs('+0')=",                    (string)NumPower::abs('+0'),                     "\n";
echo "abs('-0')=",                    (string)NumPower::abs('-0'),                     "\n";

/* ── '.5' and '5.' should both parse as fp128 ──────────────────────── */
echo "abs('.5')=", (string)NumPower::abs('.5'), "\n";
echo "abs('5.')=", (string)NumPower::abs('5.'), "\n";

/* ── Exponent forms ─────────────────────────────────────────────────── */
echo "abs('1e10')=",    (string)NumPower::abs('1e10'),    "\n";
echo "abs('1E10')=",    (string)NumPower::abs('1E10'),    "\n";
echo "abs('1.5e+10')=", (string)NumPower::abs('1.5e+10'), "\n";
echo "abs('1.5e-2')=",  (string)NumPower::abs('1.5e-2'),  "\n";

/* ── Whitespace at boundaries (mirrors strtod) ──────────────────────── */
echo "abs(' 1.5 ')=",       (string)NumPower::abs(' 1.5 '),       "\n";
echo "abs(\"\\t1.5\\n\")=", (string)NumPower::abs("\t1.5\n"),     "\n";

/* ── Strict validation applies uniformly across all 18 affected ops ── */
foreach (['abs','negative','positive','reciprocal','sign','sqrt','rsqrt',
          'square','sinc','exp','exp2','expm1','log','log1p','log2','log10',
          'logb'] as $op) {
    expect_throw("$op('0xff')",  fn() => NumPower::$op('0xff'),  'malformed literal');
    expect_throw("$op('abc')",   fn() => NumPower::$op('abc'),   'malformed literal');
}
expect_throw("clip('0xff',0,1)", fn() => NumPower::clip('0xff', 0, 1), 'malformed literal');

echo "DONE\n";
?>
--EXPECTF--
OK abs('') throws
OK abs('   ') throws
OK abs("\t\n") throws
OK clip('',0,1) throws
OK abs('0xff') throws
OK abs('abc') throws
OK abs('1.5.5') throws
OK abs('1.5a') throws
OK abs('1,5') throws
OK abs('1.5e') throws
OK abs('+') throws
OK abs('-') throws
OK abs('.') throws
OK abs('  -  1.5') throws
OK abs('-.') throws
OK abs('1.5e+') throws
OK abs('1.5E-') throws
OK abs('100xyz') throws
OK abs('inf!') throws
OK abs('nanjunk') throws
abs('inf')=inf
abs('Inf')=inf
abs('INF')=inf
abs('infinity')=inf
abs('Infinity')=inf
abs('-inf')=inf
abs('nan')[lower3]=nan
abs('NaN')[lower3]=nan
abs('99999999999999999999') → fp128 ~1e20: OK
abs('18446744073709551616')=18446744073709551616
abs('100000000000000000000')=100000000000000000000
abs('1e30') → fp128 ~1e30: OK
abs('9223372036854775807')=9223372036854775807
abs('9223372036854775808')=9223372036854775808
abs('18446744073709551615')=18446744073709551615
abs('-9223372036854775808')=-9223372036854775808
abs('+100')=100
abs('00000000000000000100')=100
abs('+0')=0
abs('-0')=0
abs('.5')=0.5
abs('5.')=5
abs('1e10')=10000000000
abs('1E10')=10000000000
abs('1.5e+10')=15000000000
abs('1.5e-2')=0.015
abs(' 1.5 ')=1.5
abs("\t1.5\n")=1.5
OK abs('0xff') throws
OK abs('abc') throws
OK negative('0xff') throws
OK negative('abc') throws
OK positive('0xff') throws
OK positive('abc') throws
OK reciprocal('0xff') throws
OK reciprocal('abc') throws
OK sign('0xff') throws
OK sign('abc') throws
OK sqrt('0xff') throws
OK sqrt('abc') throws
OK rsqrt('0xff') throws
OK rsqrt('abc') throws
OK square('0xff') throws
OK square('abc') throws
OK sinc('0xff') throws
OK sinc('abc') throws
OK exp('0xff') throws
OK exp('abc') throws
OK exp2('0xff') throws
OK exp2('abc') throws
OK expm1('0xff') throws
OK expm1('abc') throws
OK log('0xff') throws
OK log('abc') throws
OK log1p('0xff') throws
OK log1p('abc') throws
OK log2('0xff') throws
OK log2('abc') throws
OK log10('0xff') throws
OK log10('abc') throws
OK logb('0xff') throws
OK logb('abc') throws
OK clip('0xff',0,1) throws
DONE
