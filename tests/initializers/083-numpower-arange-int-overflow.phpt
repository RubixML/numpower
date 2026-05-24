--TEST--
NumPower::arange() handles int64 / uint64 length-computation boundaries without overflow
--FILE--
<?php
/* Pre-existing bug: arange_length_int64 used signed arithmetic on
 * `stop - start`, `-step`, and `(diff + step - 1)` which all overflow
 * at the int64 boundary. arange_length_uint64 had a matching wrap on
 * `(diff + step - 1)` near UINT64_MAX. The fix routes the magnitude
 * through `uint64_t` and uses the overflow-safe ceiling-division
 * identity `(udiff - 1) / abs_step + 1`. This test pins the boundary
 * behaviour so a regression to the old formulation surfaces here. */

/* (stop - start) > LONG_MAX → length wouldn't fit in `long` → clean
   error instead of silent UB. */
try {
    NumPower::arange('9223372036854775807', '-9223372036854775808', 1, 'int64');
    echo "BAD: no throw on huge i64 range\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'step must be non-zero')
        ? "huge_i64_range: OK\n"
        : "huge_i64_range: BAD ({$e->getMessage()})\n";
}

/* step == INT64_MIN — `-step` was UB before the fix. */
$a = NumPower::arange('-100', '0', '-9223372036854775808', 'int64');
echo 'i64_min_step: shape=', json_encode($a->shape()),
     ' val=', (string)$a, "\n";

/* Same but for uint64: diff > LONG_MAX → reject. */
try {
    NumPower::arange('18446744073709551615', '0', '1', 'uint64');
    echo "BAD: no throw on huge u64 range\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'step must be non-zero')
        ? "huge_u64_range: OK\n"
        : "huge_u64_range: BAD ({$e->getMessage()})\n";
}

/* Small near-boundary slices must still produce correct values. */
$a = NumPower::arange('9223372036854775807', '9223372036854775804', '1', 'int64');
echo 'i64_near_max: ', (string)$a, "\n";

$a = NumPower::arange('-9223372036854775806', '-9223372036854775808', '1', 'int64');
echo 'i64_near_min: ', (string)$a, "\n";

$a = NumPower::arange('18446744073709551615', '18446744073709551612', '1', 'uint64');
echo 'u64_near_max: ', (string)$a, "\n";

/* Negative-step sequence near INT64_MIN — `(uint64_t)start - (uint64_t)stop`
   was the previously-UB subtraction. */
$a = NumPower::arange('-9223372036854775808', '-9223372036854775805', '-1', 'int64');
echo 'i64_neg_step_near_min: ', (string)$a, "\n";
?>
--EXPECT--
huge_i64_range: OK
i64_min_step: shape=[1] val=[0]
huge_u64_range: OK
i64_near_max: [9223372036854775804, 9223372036854775805, 9223372036854775806]
i64_near_min: [-9223372036854775808, -9223372036854775807]
u64_near_max: [18446744073709551612, 18446744073709551613, 18446744073709551614]
i64_neg_step_near_min: [-9223372036854775805, -9223372036854775806, -9223372036854775807]
