--TEST--
NumPower::poisson() raises clear errors on bad scalars, invalid dtype/device
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine in a sane state. */

try {
    NumPower::poisson([4], 1.0, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::poisson([4], 1.0, 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* Negative lam rejected with a clear message. */
foreach ([-0.1, -5.0, '-1'] as $bad_lam) {
    try {
        NumPower::poisson([4], $bad_lam);
        echo "BAD: no throw on lam=$bad_lam\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'non-negative')
            ? "neg_lam_" . str_replace('.', '_', (string)$bad_lam) . ": OK\n"
            : "neg_lam_$bad_lam: BAD ({$e->getMessage()})\n";
    }
}

/* NaN lam rejected. */
try {
    NumPower::poisson([4], NAN);
    echo "BAD: no throw on NaN lam\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'non-negative')
        ? "nan_lam: OK\n"
        : "nan_lam: BAD ({$e->getMessage()})\n";
}

/* Non-numeric scalar inputs rejected per arg. */
try {
    NumPower::poisson([4], new stdClass());
    echo "BAD: no throw on object lam\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'must be int, float, or numeric string')
        ? "obj_lam: OK\n"
        : "obj_lam: BAD ({$e->getMessage()})\n";
}

try {
    NumPower::poisson([4], [1, 2]);
    echo "BAD: no throw on array lam\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'must be int, float, or numeric string')
        ? "arr_lam: OK\n"
        : "arr_lam: BAD ({$e->getMessage()})\n";
}

/* Negative shape entries rejected by ndarray_parse_typed_shape. */
try {
    NumPower::poisson([-1]);
    echo "BAD: no throw on negative shape\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'negative dimensions')
        ? "neg_shape: OK\n"
        : "neg_shape: BAD ({$e->getMessage()})\n";
}

/* dtype passed as an array — Z_PARAM_STRING rejects with TypeError. */
try {
    NumPower::poisson([4], 1.0, ['float32']);
    echo "BAD: no throw on array dtype\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_dtype: OK\n"
        : "array_dtype: BAD ({$e->getMessage()})\n";
}

/* device passed as an array — Z_PARAM_LONG rejects with TypeError. */
try {
    NumPower::poisson([4], 1.0, 'float32', ['gpu']);
    echo "BAD: no throw on array device\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_device: OK\n"
        : "array_device: BAD ({$e->getMessage()})\n";
}

/* After every failure the engine still produces a correct poisson. */
$ok = NumPower::poisson([4], 1.0, 'float32');
echo 'recovered: ', ($ok->shape() === [4] ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
neg_lam_-0_1: OK
neg_lam_-5: OK
neg_lam_-1: OK
nan_lam: OK
obj_lam: OK
arr_lam: OK
neg_shape: OK
array_dtype: OK
array_device: OK
recovered: OK
