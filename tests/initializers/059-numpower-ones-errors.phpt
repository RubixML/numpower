--TEST--
NumPower::ones() raises clear errors on invalid dtype, invalid device, and negative dimensions
--FILE--
<?php
/* Every branch of the parameter validation must throw a catchable Error.
   The message text is part of the public contract (tooling greps the dtype
   list and the device-domain hint). After each failure the engine must
   remain usable. */

try {
    NumPower::ones([2], 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype_err: OK\n"
        : "dtype_err: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::ones([2], 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), "Invalid device")
            ? "dev_err_$bad: OK\n"
            : "dev_err_$bad: BAD ({$e->getMessage()})\n";
    }
}

foreach ([[-1], [-5], [3, -2], [4, 4, -1]] as $shape) {
    $tag = 'neg_dim ' . json_encode($shape);
    try {
        NumPower::ones($shape);
        echo "BAD: no throw on shape=", json_encode($shape), "\n";
    } catch (\Error $e) {
        $msg = $e->getMessage() === 'negative dimensions are not allowed'
            ? "OK"
            : "BAD ({$e->getMessage()})";
        echo $tag, ': ', $msg, "\n";
    }
}

$ok = NumPower::ones([3, 3]);
echo 'recovered: ', ((string)NumPower::sum($ok) === '9' ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
dtype_err: OK
dev_err_2: OK
dev_err_-1: OK
dev_err_99: OK
neg_dim [-1]: OK
neg_dim [-5]: OK
neg_dim [3,-2]: OK
neg_dim [4,4,-1]: OK
recovered: OK
