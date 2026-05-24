--TEST--
NumPower::diag() raises clear errors on invalid rank, dtype, or device
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine sane. */

/* 0-D input — rejected as not 1-D and not 2-D. */
try {
    $s = new NDArray(5.0);
    NumPower::diag($s);
    echo "BAD: no throw on 0-D input\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'input must be 1-D or 2-D')
        ? "scalar: OK\n"
        : "scalar: BAD ({$e->getMessage()})\n";
}

/* 3-D input — rejected. */
try {
    NumPower::diag(NumPower::array([[[1,2],[3,4]],[[5,6],[7,8]]]));
    echo "BAD: no throw on 3-D input\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'input must be 1-D or 2-D')
        ? "3d: OK\n"
        : "3d: BAD ({$e->getMessage()})\n";
}

/* Invalid dtype. */
try {
    NumPower::diag(NumPower::array([1, 2, 3]), 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

/* Invalid device. */
foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::diag(NumPower::array([1, 2, 3]), 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* After every failure the engine still produces a correct diag. */
$d = NumPower::diag(NumPower::array([1, 2, 3]));
echo 'recovered: ',
     ((string)NumPower::sum($d) === '6' ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
scalar: OK
3d: OK
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
recovered: OK
