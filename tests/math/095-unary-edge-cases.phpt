--TEST--
NumPower unary ops: 0-D scalars, empty arrays, NaN / Inf, dimension-exceeding sizes
--FILE--
<?php
/* Edge cases for the typed unary dispatcher:
   - 0-D input collapses to a PHP scalar of the result dtype;
   - empty (zero-size) input returns an empty NDArray of the result dtype;
   - NaN / Inf propagate exactly per IEEE 754;
   - sizes larger than a single block exercise the multi-block path. */

/* 0-D input ─────────────────────────────────────────────────────────── */
echo NumPower::abs(NumPower::array(-3.5)), "\n";          /* 3.5  */
echo NumPower::sign(NumPower::array(-5.0)), "\n";         /* -1   */
echo NumPower::sqrt(NumPower::array(9.0)), "\n";          /* 3    */
echo NumPower::square(NumPower::array(-4.0)), "\n";       /* 16   */
echo NumPower::clip(NumPower::array(10), 0, 5), "\n";     /* 5    */

/* int dtype preserved end-to-end: abs of a 1-D int32 array remains
   int32 (verified by offset-access returning a PHP int). */
$x = NumPower::abs(NumPower::array([-7], 'int32'))[0];
if (is_int($x) && $x === 7) echo "OK abs preserves int dtype\n";
else                         echo "FAIL abs int dtype: ", var_export($x, true), "\n";

/* fp128 0-D → string scalar */
$y = NumPower::abs(NumPower::array(['-1.5'], 'float128'));  /* still a 1-D NDArray of length 1 */
print_r($y->toArray());

/* Empty (zero-size) input ───────────────────────────────────────────── */
$emp = NumPower::abs(NumPower::array([], 'float32'));
echo "empty count: ", count($emp), "\n";       /* 0 */
print_r($emp->shape());                         /* [0] */

/* NaN / Inf propagation ─────────────────────────────────────────────── */
$nan_in = NumPower::array([NAN, INF, -INF], 'float64');
$abs_nan = NumPower::abs($nan_in)->toArray();
echo "abs(NaN) is NaN: ",   is_nan($abs_nan[0])      ? "yes" : "no", "\n";
echo "abs(+Inf): ",          is_infinite($abs_nan[1]) && $abs_nan[1] > 0 ? "+Inf" : "?", "\n";
echo "abs(-Inf): ",          is_infinite($abs_nan[2]) && $abs_nan[2] > 0 ? "+Inf" : "?", "\n";

$sign_nan = NumPower::sign($nan_in)->toArray();
echo "sign(+Inf): ", $sign_nan[1], "\n";
echo "sign(-Inf): ", $sign_nan[2], "\n";

$sqrt_neg = NumPower::sqrt(NumPower::array([-1.0, 0.0, 4.0], 'float64'))->toArray();
echo "sqrt(-1) is NaN: ", (is_nan($sqrt_neg[0]) ? "yes" : "no"), "\n";
echo "sqrt(0): ", $sqrt_neg[1], ", sqrt(4): ", $sqrt_neg[2], "\n";

$rec_zero = NumPower::reciprocal(NumPower::array([0.0, 1.0], 'float64'))->toArray();
echo "reciprocal(0) is INF: ", is_infinite($rec_zero[0]) ? "yes" : "no", "\n";

/* sinc continuity at zero ───────────────────────────────────────────── */
$sinc0 = NumPower::sinc(NumPower::array([0.0], 'float64'))->toArray();
if (abs($sinc0[0] - 1.0) < 1e-15) echo "OK sinc(0) == 1\n";
else                               echo "FAIL sinc(0): ", $sinc0[0], "\n";

/* Large dimension to cover multi-block kernel launch ────────────────── */
$big = [];
for ($i = 0; $i < 4096; $i++) $big[] = $i;
$big_arr = NumPower::array($big, 'float64');
$sq = NumPower::square($big_arr);
$sq_back = $sq->toArray();
$ok = true;
foreach ([0, 1, 100, 1000, 4095] as $i) {
    if (abs($sq_back[$i] - ($i * $i)) > 1e-9) {
        echo "FAIL big square[$i]: got=", $sq_back[$i], " want=", $i*$i, "\n";
        $ok = false;
    }
}
if ($ok) echo "OK 4096-element square\n";

/* 4-D array shape preservation. */
$rows = [];
for ($i = 0; $i < 24; $i++) $rows[] = ($i % 5) - 2;
$nd4 = NumPower::array([[[$rows[0],$rows[1],$rows[2]],
                         [$rows[3],$rows[4],$rows[5]]],
                        [[$rows[6],$rows[7],$rows[8]],
                         [$rows[9],$rows[10],$rows[11]]]], 'int16');
$abs4 = NumPower::abs($nd4);
print_r($abs4->shape());          /* [2,2,3] */

echo "DONE\n";
?>
--EXPECTF--
3.5
-1
3
16
5
OK abs preserves int dtype
Array
(
    [0] => 1.5
)
empty count: 0
Array
(
    [0] => 0
)
abs(NaN) is NaN: yes
abs(+Inf): +Inf
abs(-Inf): +Inf
sign(+Inf): 1
sign(-Inf): -1
sqrt(-1) is NaN: yes
sqrt(0): 0, sqrt(4): 2
reciprocal(0) is INF: yes
OK sinc(0) == 1
OK 4096-element square
Array
(
    [0] => 2
    [1] => 2
    [2] => 3
)
DONE
