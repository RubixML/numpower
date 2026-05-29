--TEST--
clip() bound saturation for integer dtypes + negative-magnitude string overflow → fp128 escalation
--FILE--
<?php
/* Two pre-existing bugs surfaced by the second review pass and fixed
   on this branch.

   1. clip() on unsigned-int dtypes silently wrapped negative bounds.
      Test: clip(uint8 [10, 50, 100, 200], -50, 100) returned
      [100, 100, 100, 100] instead of [10, 50, 100, 100]. Root cause:
      `unary_parse_typed_scalar` routed "-50" through
      `ndarray_set_from_string` -> `(uint8)strtoull(-50)` = 206, then
      the clip kernel saw `lo (206) > hi (100)` and collapsed every
      element to `hi`. Fix: saturate the bound to the dtype's range
      (PyTorch / NumPy clamp contract): negative-on-unsigned -> 0,
      out-of-range positive -> dtype max, out-of-range negative on
      signed -> dtype min.

   2. ndarray_infer_dtype_from_string() classified negative-magnitude
      literals above |INT64_MIN| as int64 unconditionally, hiding the
      INT64_MIN saturation that strtoll would apply inside
      `ndarray_set_from_string`. Test: abs("-18446744073709551615")
      silently produced INT64_MIN. Fix: digit-by-digit magnitude check
      against |INT64_MIN| ("9223372036854775808"); larger negatives
      escalate to float128.
*/

function check_arr($label, $got, $want) {
    if ($got === $want) { echo "OK $label\n"; return; }
    echo "FAIL $label: got=", json_encode($got), " want=", json_encode($want), "\n";
}

function check_type($label, $got, $type) {
    $actual = gettype($got);
    if ($actual === $type) { echo "OK $label is $type\n"; return; }
    echo "FAIL $label: got type=$actual want=$type, value=", var_export($got, true), "\n";
}

/* ── 1. clip bound saturation, every integer dtype × CPU and GPU ─────── */

/* uint8 / uint16 / uint32 / uint64: negative `min` saturates to 0. */
foreach (['uint8' => false, 'uint16' => false,
          'uint32' => false, 'uint64' => true] as $dt => $is_wide) {
    $a = NumPower::array([10, 50, 100, 200], $dt);
    /* uint64's toArray() returns strings (project contract for wide
       dtypes); the narrow uints return ints. Expected output accordingly. */
    $expected = $is_wide ? ['10','50','100','100'] : [10, 50, 100, 100];
    $r = NumPower::clip($a, -50, 100)->toArray();
    check_arr("clip($dt, -50, 100) CPU", $r, $expected);
    $r = NumPower::clip($a->gpu(), -50, 100)->cpu()->toArray();
    check_arr("clip($dt, -50, 100) GPU", $r, $expected);
}

/* Over-max bound: `max` greater than dtype's MAX saturates to MAX. */
$a = NumPower::array([10, 50, 100, 200], 'uint8');
check_arr("clip(uint8, 0, 300) CPU",
          NumPower::clip($a, 0, 300)->toArray(),
          [10, 50, 100, 200]);
check_arr("clip(uint8, 0, 1000000) CPU",
          NumPower::clip($a, 0, 1000000)->toArray(),
          [10, 50, 100, 200]);

/* Signed-int saturation: -300 on int8 → -128 (INT8_MIN); 200 → 127. */
$a = NumPower::array([-100, 50, 100], 'int8');
check_arr("clip(int8, -300, 200) CPU",
          NumPower::clip($a, -300, 200)->toArray(),
          [-100, 50, 100]);
check_arr("clip(int8, -10, 30) CPU",
          NumPower::clip($a, -10, 30)->toArray(),
          [-10, 30, 30]);

/* String bounds — same saturation. */
check_arr("clip(uint8, '-50', '100') CPU",
          NumPower::clip(NumPower::array([10, 50, 100, 200], 'uint8'), '-50', '100')->toArray(),
          [10, 50, 100, 100]);

/* int64 / uint64 wide bounds: ERANGE saturation from strtoll/strtoull.
   int64's toArray returns PHP int (long), uint64 returns string. */
$a = new NDArray(['0', '1000000000000000000', '9223372036854775807'], 'int64');
$r = NumPower::clip($a, '-99999999999999999999', '99999999999999999999')->toArray();
check_arr("clip(int64, very-wide bounds) CPU",
          $r, [0, 1000000000000000000, 9223372036854775807]);

$a = new NDArray(['0', '18446744073709551614', '18446744073709551615'], 'uint64');
$r = NumPower::clip($a, '-1', '99999999999999999999')->toArray();
check_arr("clip(uint64, negative min) CPU",
          $r,
          ['0', '18446744073709551614', '18446744073709551615']);

/* float dtypes are unaffected: -inf / +inf bounds clamp correctly. */
$a = NumPower::array([-1e10, 0.0, 1e10], 'float64');
$r = NumPower::clip($a, -100.0, 100.0)->toArray();
check_arr("clip(float64) CPU", $r, [-100.0, 0.0, 100.0]);

/* ── 2. Negative-magnitude string overflow → float128 escalation ────── */

/* INT64_MIN exact fit → int64. */
check_type("abs('-9223372036854775808')", NumPower::abs('-9223372036854775808'), 'integer');

/* |INT64_MIN| + 1 → float128 (escalated; no silent saturation). */
check_type("abs('-9223372036854775809')", NumPower::abs('-9223372036854775809'), 'string');
echo "abs('-9223372036854775809') = ",
     NumPower::abs('-9223372036854775809'), "\n";

/* Way past int64 range → float128. */
check_type("abs('-18446744073709551615')", NumPower::abs('-18446744073709551615'), 'string');
echo "abs('-18446744073709551615') = ",
     NumPower::abs('-18446744073709551615'), "\n";

/* The escalation applies uniformly across the affected ops. */
foreach (['negative', 'positive', 'sign', 'square'] as $op) {
    $r = NumPower::$op('-99999999999999999999');
    check_type("$op('-99999999999999999999')", $r, 'string');
}

/* sqrt/log/exp on a negative giant fp128 magnitude: routes to fp128
   compute. sqrt of negative → nan; log of negative → nan. */
echo "sqrt('-99999999999999999999') = ",
     NumPower::sqrt('-99999999999999999999'), "\n";
echo "log('-99999999999999999999') = ",
     NumPower::log('-99999999999999999999'), "\n";

echo "DONE\n";
?>
--EXPECT--
OK clip(uint8, -50, 100) CPU
OK clip(uint8, -50, 100) GPU
OK clip(uint16, -50, 100) CPU
OK clip(uint16, -50, 100) GPU
OK clip(uint32, -50, 100) CPU
OK clip(uint32, -50, 100) GPU
OK clip(uint64, -50, 100) CPU
OK clip(uint64, -50, 100) GPU
OK clip(uint8, 0, 300) CPU
OK clip(uint8, 0, 1000000) CPU
OK clip(int8, -300, 200) CPU
OK clip(int8, -10, 30) CPU
OK clip(uint8, '-50', '100') CPU
OK clip(int64, very-wide bounds) CPU
OK clip(uint64, negative min) CPU
OK clip(float64) CPU
OK abs('-9223372036854775808') is integer
OK abs('-9223372036854775809') is string
abs('-9223372036854775809') = 9223372036854775809
OK abs('-18446744073709551615') is string
abs('-18446744073709551615') = 18446744073709551615
OK negative('-99999999999999999999') is string
OK positive('-99999999999999999999') is string
OK sign('-99999999999999999999') is string
OK square('-99999999999999999999') is string
sqrt('-99999999999999999999') = nan
log('-99999999999999999999') = nan
DONE
