--TEST--
NumPower unary ops: bare-string scalars throw a useful error; clip accepts int / float / string bounds
--FILE--
<?php
/* Bare-string scalar input has no peer dtype to anchor on, so the
   dispatcher throws a clear error pointing the caller at the correct
   string-intake path. clip's min / max accept int / float / string
   so fp128 / uint64 callers can pass loss-free bounds. */

function expect_throw($label, $thunk) {
    try {
        $thunk();
        echo "FAIL $label: expected an error\n";
    } catch (\Error $e) {
        echo "OK $label: ", $e->getMessage(), "\n";
    }
}

expect_throw("abs('1.5')",   fn() => NumPower::abs('1.5'));
expect_throw("sqrt('4')",    fn() => NumPower::sqrt('4'));
expect_throw("clip('5',0,1)",fn() => NumPower::clip('5', 0, 1));

/* The new strict validator rejects malformed numeric bounds (was silent
   `strtod` → 0 before this branch). */
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
OK abs('1.5'): Cannot infer dtype for a bare string.%s
OK sqrt('4'): Cannot infer dtype for a bare string.%s
OK clip('5',0,1): Cannot infer dtype for a bare string.%s
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
