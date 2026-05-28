--TEST--
NumPower unary ops: regressions for pre-existing bugs (rsqrt → arccos; positive → abs; sqrt/square legacy float32-only)
--FILE--
<?php
/* Three bugs the typed-unary dispatcher fixes:
   1) `NumPower::rsqrt` on GPU used to call `cuda_float_arccos` (copy/paste
      from a neighbouring method). Result was the inverse cosine, not
      1/sqrt(x). The CPU path was OK but produced float32 regardless of
      dtype.
   2) `NumPower::positive` mapped through `float_positive`, defined as
      `val < 0 ? -val : val` — i.e. it was an `abs` clone, not the
      identity NumPy's `np.positive` promises.
   3) Every unary op went through `NDArray_Map` (always float32 output)
      or `NDArrayMathGPU_ElementWise` (float32 input via NDArray_F32DATA).
      Non-float32 inputs read/wrote garbage. */

/* Bug 1: rsqrt is 1/sqrt(x), not arccos(x). */
$r = NumPower::rsqrt(NumPower::array([1.0, 4.0, 9.0], 'float32'));
$out = $r->toArray();
$want = [1.0, 0.5, 1.0/3];
$ok = true;
for ($i = 0; $i < 3; $i++) {
    if (abs($out[$i] - $want[$i]) > 1e-5) {
        echo "FAIL rsqrt[$i]: got=", $out[$i], " want=", $want[$i], "\n"; $ok = false;
    }
}
if ($ok) echo "OK rsqrt is 1/sqrt(x), not arccos\n";

/* Bug 2: positive is identity, not abs. */
$p = NumPower::positive(NumPower::array([-3.0, 0.0, 5.0]));
$ok = true;
$want = [-3.0, 0.0, 5.0];
foreach ($p->toArray() as $i => $v)
    if (abs($v - $want[$i]) > 1e-12) { echo "FAIL positive[$i]: got=$v want={$want[$i]}\n"; $ok = false; }
if ($ok) echo "OK positive is identity (not abs)\n";

/* Bug 3: dtype is preserved for non-float32 inputs.

   Before the refactor:
     - `NumPower::abs(NumPower::array([1.0], 'float64'))` returned a
       float64 (via the special-case NDArray_Abs branch) — coincidence
       only abs had,
     - every other op (negate, sign, square, sqrt, rsqrt, recip, sinc,
       positive, clip) returned float32 regardless of source dtype,
       silently downgrading every fp64 / fp128 / wide-int input.

   The typed dispatcher restores per-dtype output for the preserving
   ops and per-dtype-aware promotion for the floating-only ops. */

/* float64 preserves dtype on every op (check via toArray() string form
   on a large value beyond float32 precision). */
$big_f64 = NumPower::array([1.0e15 + 1.0, -1.0e15 - 1.0], 'float64');
$abs_f64 = NumPower::abs($big_f64)->toArray();
if ($abs_f64[0] === 1.0e15 + 1.0 && $abs_f64[1] === 1.0e15 + 1.0)
    echo "OK abs preserves float64 precision\n";
else
    echo "FAIL abs f64 precision: ", json_encode($abs_f64), "\n";

$neg_f64 = NumPower::negative($big_f64)->toArray();
if ($neg_f64[0] === -1.0e15 - 1.0 && $neg_f64[1] === 1.0e15 + 1.0)
    echo "OK negative preserves float64 precision\n";
else
    echo "FAIL negative f64 precision: ", json_encode($neg_f64), "\n";

$sq_f64 = NumPower::square(NumPower::array([1.0e7, -1.0e7 + 0.5], 'float64'))->toArray();
/* (1e7+0.5)^2 = 1e14 + 1e7 + 0.25, distinguishable in float64 but not float32. */
if (abs($sq_f64[1] - (1.0e14 - 1.0e7 + 0.25)) < 100)
    echo "OK square preserves float64 precision\n";
else
    echo "FAIL square f64: got=", $sq_f64[1], "\n";

/* fp128: 30-digit precision survives end-to-end.
   libquadmath outputs the expanded form '1234567890...01.23',
   DD emulation outputs scientific '1.2345...012e+30'; strip
   non-digits so we compare the underlying digit sequence. */
$f128 = NumPower::array(['1.23456789012345678901234567890123e30'], 'float128');
$abs_f128 = NumPower::abs($f128)->toArray();
$exp = '1234567890123456789012345678901';   /* 31 sig-digits */
$digits_only = preg_replace('/[^0-9]/', '', $abs_f128[0]);
if (str_starts_with($digits_only, substr($exp, 0, 30)))
    echo "OK abs preserves fp128 precision\n";
else
    echo "FAIL abs fp128: got=", $abs_f128[0], "\n";

/* uint64 max preserved through abs / square / sign / negate. */
$u = NumPower::array(['18446744073709551615'], 'uint64');
$abs_u = NumPower::abs($u)->toArray();
if ($abs_u[0] === '18446744073709551615') echo "OK abs preserves uint64 precision\n";
else                                       echo "FAIL abs u64: got=", $abs_u[0], "\n";

$sign_u = NumPower::sign($u)->toArray();
if ($sign_u[0] === '1') echo "OK sign uint64 returns 1 for non-zero\n";
else                     echo "FAIL sign u64: got=", $sign_u[0], "\n";

echo "DONE\n";
?>
--EXPECT--
OK rsqrt is 1/sqrt(x), not arccos
OK positive is identity (not abs)
OK abs preserves float64 precision
OK negative preserves float64 precision
OK square preserves float64 precision
OK abs preserves fp128 precision
OK abs preserves uint64 precision
OK sign uint64 returns 1 for non-zero
DONE
