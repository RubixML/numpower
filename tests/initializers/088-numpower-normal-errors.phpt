--TEST--
NumPower::normal() raises clear errors on bad scalars, invalid dtype/device
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine in a sane state. */

try {
    NumPower::normal([4], 0, 1, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::normal([4], 0, 1, 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* Non-numeric scalar inputs are rejected per arg. */
foreach (['loc' => [new stdClass(), 1],
          'scale' => [0, [1, 2]]] as $tag => [$lc, $sc]) {
    try {
        NumPower::normal([4], $lc, $sc);
        echo "BAD: no throw on bad $tag\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'must be int, float, or numeric string')
            ? "bad_$tag: OK\n"
            : "bad_$tag: BAD ({$e->getMessage()})\n";
    }
}

/* uint64 with a negative scalar arg rejects with a clear message. */
foreach (['loc' => [-1, 1], 'scale' => [0, -1]] as $tag => [$lc, $sc]) {
    try {
        NumPower::normal([4], $lc, $sc, 'uint64');
        echo "BAD: no throw on uint64 negative $tag\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'non-negative for uint64')
            ? "u64_neg_$tag: OK\n"
            : "u64_neg_$tag: BAD ({$e->getMessage()})\n";
    }
}

/* Negative scale rejected for all DOUBLE-kind dtypes (mathematically a
   stddev cannot be negative; cuRAND also documents `stddev > 0`). */
foreach ([-0.1, -1.0] as $bad_scale) {
    try {
        NumPower::normal([4], 0.0, $bad_scale, 'float32');
        echo "BAD: no throw on neg scale=$bad_scale\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'scale must be non-negative')
            ? "neg_scale_" . str_replace(['.', '-'], ['_', 'm'], (string)$bad_scale) . ": OK\n"
            : "neg_scale_$bad_scale: BAD ({$e->getMessage()})\n";
    }
}

/* NaN scale rejected too. */
try {
    NumPower::normal([4], 0.0, NAN);
    echo "BAD: no throw on NaN scale\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'scale must be non-negative')
        ? "nan_scale: OK\n"
        : "nan_scale: BAD ({$e->getMessage()})\n";
}

/* Negative fp128 scale rejected. */
try {
    NumPower::normal([4], '0.0', '-1.0', 'float128');
    echo "BAD: no throw on neg fp128 scale\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'scale must be non-negative')
        ? "fp128_neg_scale: OK\n"
        : "fp128_neg_scale: BAD ({$e->getMessage()})\n";
}

/* scale = 0 is permitted (degenerate distribution; every sample = loc). */
$ok_zero_scale = NumPower::normal([8], 7.0, 0.0, 'float32');
$all_loc = true;
foreach ($ok_zero_scale->toArray() as $v) {
    if ((float)$v !== 7.0) { $all_loc = false; break; }
}
echo 'scale_zero_permitted: ', ($all_loc ? 'OK' : 'BAD'), "\n";

/* Negative shape entries are rejected by ndarray_parse_typed_shape. */
try {
    NumPower::normal([-1], 0, 1, 'float32');
    echo "BAD: no throw on negative shape\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'negative dimensions')
        ? "neg_shape: OK\n"
        : "neg_shape: BAD ({$e->getMessage()})\n";
}

/* After every failure the engine still produces a correct normal. */
$ok = NumPower::normal([4], 0.0, 1.0, 'float32');
echo 'recovered: ', ($ok->shape() === [4] ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
bad_loc: OK
bad_scale: OK
u64_neg_loc: OK
u64_neg_scale: OK
neg_scale_m0_1: OK
neg_scale_m1: OK
nan_scale: OK
fp128_neg_scale: OK
scale_zero_permitted: OK
neg_shape: OK
recovered: OK
