--TEST--
NumPower::uniform() raises clear errors on bad scalars, invalid dtype/device
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine in a sane state. */

try {
    NumPower::uniform([4], 0, 1, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::uniform([4], 0, 1, 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* Non-numeric scalar inputs are rejected per arg with a clear message. */
foreach (['low' => [new stdClass(), 1],
          'high' => [0, [1, 2]]] as $tag => [$lc, $sc]) {
    try {
        NumPower::uniform([4], $lc, $sc);
        echo "BAD: no throw on bad $tag\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'must be int, float, or numeric string')
            ? "bad_$tag: OK\n"
            : "bad_$tag: BAD ({$e->getMessage()})\n";
    }
}

/* uint64 with a negative scalar arg rejects with a clear message. */
foreach (['low' => [-1, 1], 'high' => [0, -1]] as $tag => [$lc, $sc]) {
    try {
        NumPower::uniform([4], $lc, $sc, 'uint64');
        echo "BAD: no throw on uint64 negative $tag\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'non-negative for uint64')
            ? "u64_neg_$tag: OK\n"
            : "u64_neg_$tag: BAD ({$e->getMessage()})\n";
    }
}

/* Negative shape entries are rejected by ndarray_parse_typed_shape. */
try {
    NumPower::uniform([-1], 0, 1, 'float32');
    echo "BAD: no throw on negative shape\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'negative dimensions')
        ? "neg_shape: OK\n"
        : "neg_shape: BAD ({$e->getMessage()})\n";
}

/* dtype passed as an array — Z_PARAM_STRING rejects with TypeError. */
try {
    NumPower::uniform([4], 0, 1, ['float32']);
    echo "BAD: no throw on array dtype\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_dtype: OK\n"
        : "array_dtype: BAD ({$e->getMessage()})\n";
}

/* device passed as an array — Z_PARAM_LONG rejects with TypeError. */
try {
    NumPower::uniform([4], 0, 1, 'float32', ['gpu']);
    echo "BAD: no throw on array device\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_device: OK\n"
        : "array_device: BAD ({$e->getMessage()})\n";
}

/* After every failure the engine still produces a correct uniform. */
$ok = NumPower::uniform([4], 0.0, 1.0, 'float32');
echo 'recovered: ', ($ok->shape() === [4] ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
bad_low: OK
bad_high: OK
u64_neg_low: OK
u64_neg_high: OK
neg_shape: OK
array_dtype: OK
array_device: OK
recovered: OK
