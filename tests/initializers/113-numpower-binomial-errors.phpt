--TEST--
NumPower::randomBinomial() raises clear errors on bad n, p, dtype, or device
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine in a sane state. */

try {
    NumPower::randomBinomial([4], 10, 0.5, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::randomBinomial([4], 10, 0.5, 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* Negative n rejected. */
foreach ([-1.0, -5.0] as $bad_n) {
    try {
        NumPower::randomBinomial([4], $bad_n, 0.5);
        echo "BAD: no throw on n=$bad_n\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'non-negative')
            ? "neg_n_" . str_replace(['.', '-'], ['_', 'm'], (string)$bad_n) . ": OK\n"
            : "neg_n_$bad_n: BAD ({$e->getMessage()})\n";
    }
}

/* NaN n rejected. */
try {
    NumPower::randomBinomial([4], NAN, 0.5);
    echo "BAD: no throw on NaN n\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'non-negative')
        ? "nan_n: OK\n"
        : "nan_n: BAD ({$e->getMessage()})\n";
}

/* n > INT_MAX rejected. */
try {
    NumPower::randomBinomial([4], 1e20, 0.5);
    echo "BAD: no throw on huge n\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'INT_MAX')
        ? "huge_n: OK\n"
        : "huge_n: BAD ({$e->getMessage()})\n";
}

/* p out of [0, 1] rejected. */
foreach ([-0.001, 1.001, -1.0, 2.0] as $bad_p) {
    try {
        NumPower::randomBinomial([4], 10, $bad_p);
        echo "BAD: no throw on p=$bad_p\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'p must be in')
            ? "bad_p_" . str_replace(['.', '-'], ['_', 'm'], (string)$bad_p) . ": OK\n"
            : "bad_p_$bad_p: BAD ({$e->getMessage()})\n";
    }
}

/* NaN p rejected. */
try {
    NumPower::randomBinomial([4], 10, NAN);
    echo "BAD: no throw on NaN p\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'p must be in')
        ? "nan_p: OK\n"
        : "nan_p: BAD ({$e->getMessage()})\n";
}

/* Negative shape entries rejected. */
try {
    NumPower::randomBinomial([-1], 10, 0.5);
    echo "BAD: no throw on negative shape\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'negative dimensions')
        ? "neg_shape: OK\n"
        : "neg_shape: BAD ({$e->getMessage()})\n";
}

/* dtype passed as an array. */
try {
    NumPower::randomBinomial([4], 10, 0.5, ['float32']);
    echo "BAD: no throw on array dtype\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_dtype: OK\n"
        : "array_dtype: BAD ({$e->getMessage()})\n";
}

/* device passed as an array. */
try {
    NumPower::randomBinomial([4], 10, 0.5, 'float32', ['gpu']);
    echo "BAD: no throw on array device\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_device: OK\n"
        : "array_device: BAD ({$e->getMessage()})\n";
}

/* After every failure the engine still produces a correct sample. */
$ok = NumPower::randomBinomial([4], 10, 0.5, 'float32');
echo 'recovered: ', ($ok->shape() === [4] ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
neg_n_m1: OK
neg_n_m5: OK
nan_n: OK
huge_n: OK
bad_p_m0_001: OK
bad_p_1_001: OK
bad_p_m1: OK
bad_p_2: OK
nan_p: OK
neg_shape: OK
array_dtype: OK
array_device: OK
recovered: OK
