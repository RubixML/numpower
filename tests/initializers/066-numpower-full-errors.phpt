--TEST--
NumPower::full() raises clear errors on invalid dtype, device, dim, or fill-value type
--FILE--
<?php
/* Every validation branch must throw a catchable Error. The engine must
   remain usable after each failure (no leaked shape allocation, no
   unbalanced refcounts). */

try {
    NumPower::full([2], 1, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype_err: OK\n"
        : "dtype_err: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::full([2], 1, 'float32', $bad);
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
        NumPower::full($shape, 1);
        echo "BAD: no throw on shape=", json_encode($shape), "\n";
    } catch (\Error $e) {
        $msg = $e->getMessage() === 'negative dimensions are not allowed'
            ? "OK"
            : "BAD ({$e->getMessage()})";
        echo $tag, ': ', $msg, "\n";
    }
}

/* Unsupported fill-value types — only int/float/bool/string are accepted.
   Object, array, null are rejected. */
foreach ([
    'object' => new stdClass(),
    'array'  => [1, 2, 3],
    'null'   => null,
] as $tag => $val) {
    try {
        NumPower::full([2], $val);
        echo "BAD: no throw on $tag fill\n";
    } catch (\Error $e) {
        echo "fill_$tag: ",
             (str_starts_with($e->getMessage(), 'Invalid value type')
                 ? "OK" : "BAD ({$e->getMessage()})"),
             "\n";
    }
}

/* After every failure path the engine must still build a valid NDArray
   and produce the correct sum. */
$ok = NumPower::full([3, 3], 1, 'float32');
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
fill_object: OK
fill_array: OK
fill_null: OK
recovered: OK
