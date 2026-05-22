--TEST--
NDArray::count() returns exact axis-0 length for very large leading dimensions (no overflow)
--FILE--
<?php
/* count() returns a PHP long (zend_long, 64-bit on most platforms) but reads
   from an int* dimensions buffer. Verify that the int->long widening is
   correct for the practical range we expose (up to ~2^31-1) and that nothing
   gets truncated for large leading axes. */

$sizes = [1, 100, 10000, 100000, 1000000];

foreach ($sizes as $n) {
    /* 1-D: shape (n,). */
    $a = NumPower::zeros([$n]);
    $c = $a->count();
    echo "1D[$n]: " . ($c === $n ? "OK" : "FAIL got=$c") . PHP_EOL;
    unset($a);

    /* 2-D: shape (n, 4). */
    $a = NumPower::zeros([$n, 4]);
    $c = $a->count();
    echo "2D[$n,4]: " . ($c === $n ? "OK" : "FAIL got=$c") . PHP_EOL;
    unset($a);
}

/* return-type contract: count() must return int. */
$a = NumPower::zeros([42]);
echo "type: " . gettype($a->count()) . PHP_EOL;
echo "type-builtin: " . gettype(count($a)) . PHP_EOL;
?>
--EXPECT--
1D[1]: OK
2D[1,4]: OK
1D[100]: OK
2D[100,4]: OK
1D[10000]: OK
2D[10000,4]: OK
1D[100000]: OK
2D[100000,4]: OK
1D[1000000]: OK
2D[1000000,4]: OK
type: integer
type-builtin: integer
