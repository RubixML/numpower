--TEST--
NumPower::identity() raises clear errors on invalid size, dtype, or device
--FILE--
<?php
/* Every validation branch must throw a catchable Error. After each
   failure the engine must still produce a valid identity matrix on a
   subsequent valid call. */

foreach ([-1, -100] as $bad) {
    try {
        NumPower::identity($bad);
        echo "BAD: no throw on size=$bad\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'out of range')
            ? "size_neg_$bad: OK\n"
            : "size_neg_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* size > INT_MAX (PHP_INT_MAX = 2^63-1 on 64-bit) must be rejected
   before any allocation runs. */
try {
    NumPower::identity(PHP_INT_MAX);
    echo "BAD: no throw on size=PHP_INT_MAX\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'out of range')
        ? "size_overflow: OK\n"
        : "size_overflow: BAD ({$e->getMessage()})\n";
}

try {
    NumPower::identity(3, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype_err: OK\n"
        : "dtype_err: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::identity(3, 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_err_$bad: OK\n"
            : "dev_err_$bad: BAD ({$e->getMessage()})\n";
    }
}

$ok = NumPower::identity(3);
echo 'recovered: ', ((string)NumPower::sum($ok) === '3' ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
size_neg_-1: OK
size_neg_-100: OK
size_overflow: OK
dtype_err: OK
dev_err_2: OK
dev_err_-1: OK
dev_err_99: OK
recovered: OK
