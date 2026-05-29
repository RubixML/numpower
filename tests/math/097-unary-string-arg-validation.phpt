--TEST--
NumPower unary ops: bare-string scalars infer dtype (fp128 / int64 / uint64); clip accepts int / float / string bounds
--FILE--
<?php
/* Bare-string scalar input is now accepted by the unary dispatcher and
   by clip. The dtype is inferred from the literal:
     - decimal / exponent / inf / nan → float128 (full ~34-digit precision);
     - non-negative integer above INT64_MAX → uint64;
     - everything else → int64.
   This is the single-call intake path for fp128 / uint64 precision.
   Empty / whitespace-only strings throw a distinct error.
   clip's min / max also accept int / float / string. */

function expect_throw($label, $thunk) {
    try {
        $thunk();
        echo "FAIL $label: expected an error\n";
    } catch (\Error $e) {
        echo "OK $label: ", $e->getMessage(), "\n";
    }
}

/* ── Bare-string input now succeeds ─────────────────────────────────── */
echo "abs('1.5')=",            (string)NumPower::abs('1.5'),                 "\n";
echo "abs('-100')=",            (string)NumPower::abs('-100'),                "\n";
echo "abs('18446744073709551615')=",
                                (string)NumPower::abs('18446744073709551615'), "\n";
echo "sqrt('4')=",              (string)NumPower::sqrt('4'),                  "\n";
echo "sqrt('16.0')=",           (string)NumPower::sqrt('16.0'),               "\n";
echo "clip('5',0,1)=",          (string)NumPower::clip('5', 0, 1),            "\n";
echo "clip('5.5',0,1)=",        (string)NumPower::clip('5.5', 0, 1),          "\n";

/* ── Empty / whitespace-only / malformed strings throw ──────────────── */
expect_throw("abs('')",         fn() => NumPower::abs(''));
expect_throw("sqrt('   ')",     fn() => NumPower::sqrt('   '));
expect_throw("clip('',0,1)",    fn() => NumPower::clip('', 0, 1));
/* Stricter validation also rejects partial / illegal literals. */
expect_throw("abs('0xff')",     fn() => NumPower::abs('0xff'));
expect_throw("abs('abc')",      fn() => NumPower::abs('abc'));
expect_throw("abs('1.5.5')",    fn() => NumPower::abs('1.5.5'));
expect_throw("abs('1.5a')",     fn() => NumPower::abs('1.5a'));
expect_throw("abs('1,5')",      fn() => NumPower::abs('1,5'));
expect_throw("abs('1.5e')",     fn() => NumPower::abs('1.5e'));
expect_throw("abs('+')",        fn() => NumPower::abs('+'));
expect_throw("abs('-')",        fn() => NumPower::abs('-'));
expect_throw("abs('.')",        fn() => NumPower::abs('.'));
expect_throw("abs('  -  1.5')", fn() => NumPower::abs('  -  1.5'));

/* ── Magnitudes past UINT64_MAX escalate to float128 (not saturate) ──── */
echo "abs('99999999999999999999')=",        (string)NumPower::abs('99999999999999999999'),  "\n";
echo "abs('18446744073709551616')=",        (string)NumPower::abs('18446744073709551616'),  "\n";

/* The strict validator on clip bounds still rejects malformed numbers. */
expect_throw("clip([1],'abc','5')",  fn() => NumPower::clip([1.0], 'abc', '5'));
expect_throw("clip([1],'5','1.5e')", fn() => NumPower::clip([1.0], '5', '1.5e'));
expect_throw("clip([1],'','5')",     fn() => NumPower::clip([1.0], '', '5'));

/* clip bounds: int, float, string all accepted. */
$arr = NumPower::array([-2.0, 0.5, 3.0], 'float64');
$out = NumPower::clip($arr, -1, 2)->toArray();
echo "clip int bounds: ", json_encode($out), "\n";

$out = NumPower::clip($arr, -1.5, 2.5)->toArray();
echo "clip float bounds: ", json_encode($out), "\n";

$out = NumPower::clip($arr, '-1.0', '2.0')->toArray();
echo "clip string bounds: ", json_encode($out), "\n";

/* fp128 with full-precision string bounds. libquadmath and DD emulation
   stringify large fp128 values differently ('1e+29' vs '100...000'),
   so verify per-element with a portable form rather than print_r. */
$f = NumPower::array(['-1e30', '0', '1e30',
                       '1.23456789012345678901234567890123e29'], 'float128');
$cl = NumPower::clip($f, '0', '1e29')->toArray();
echo "fp128 clip[0]: ", ((float)$cl[0] === 0.0) ? "0" : "?{$cl[0]}", "\n";
echo "fp128 clip[1]: ", ((float)$cl[1] === 0.0) ? "0" : "?{$cl[1]}", "\n";
echo "fp128 clip[2]: ", (abs(((float)$cl[2]) / 1e29 - 1.0) < 1e-10) ? "1e29" : "?{$cl[2]}", "\n";
echo "fp128 clip[3]: ", (abs(((float)$cl[3]) / 1e29 - 1.0) < 1e-10) ? "1e29" : "?{$cl[3]}", "\n";

/* uint64 with bounds beyond PHP int range. */
$u = NumPower::array(['0', '9223372036854775807', '18446744073709551615'], 'uint64');
$cl = NumPower::clip($u, '9223372036854775806', '18446744073709551614')->toArray();
print_r($cl);

/* clip rejects non-numeric bound types. */
expect_throw("clip([],null,null)",
             fn() => NumPower::clip([1.0], null, null));

echo "DONE\n";
?>
--EXPECTF--
abs('1.5')=1.5
abs('-100')=100
abs('18446744073709551615')=18446744073709551615
sqrt('4')=2
sqrt('16.0')=4
clip('5',0,1)=1
clip('5.5',0,1)=1
OK abs(''): Numeric string expected, got an empty value.
OK sqrt('   '): Numeric string expected, got a whitespace-only value.
OK clip('',0,1): Numeric string expected, got an empty value.
OK abs('0xff'): Numeric string expected, got malformed literal: "0xff".
OK abs('abc'): Numeric string expected, got malformed literal: "abc".
OK abs('1.5.5'): Numeric string expected, got malformed literal: "1.5.5".
OK abs('1.5a'): Numeric string expected, got malformed literal: "1.5a".
OK abs('1,5'): Numeric string expected, got malformed literal: "1,5".
OK abs('1.5e'): Numeric string expected, got malformed literal: "1.5e".
OK abs('+'): Numeric string expected, got malformed literal: "+".
OK abs('-'): Numeric string expected, got malformed literal: "-".
OK abs('.'): Numeric string expected, got malformed literal: ".".
OK abs('  -  1.5'): Numeric string expected, got malformed literal: "  -  1.5".
abs('99999999999999999999')=99999999999999999999
abs('18446744073709551616')=18446744073709551616
OK clip([1],'abc','5'): NDArray clip: 'min' is not a valid number:%s
OK clip([1],'5','1.5e'): NDArray clip: 'max' has malformed exponent:%s
OK clip([1],'','5'): NDArray clip: 'min' is empty.
clip int bounds: [-1,0.5,2]
clip float bounds: [-1.5,0.5,2.5]
clip string bounds: [-1,0.5,2]
fp128 clip[0]: 0
fp128 clip[1]: 0
fp128 clip[2]: 1e29
fp128 clip[3]: 1e29
Array
(
    [0] => 9223372036854775806
    [1] => 9223372036854775807
    [2] => 18446744073709551614
)
OK clip([],null,null): NumPower::clip: 'min' must be int, float, or string.
DONE
