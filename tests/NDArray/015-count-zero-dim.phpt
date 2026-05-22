--TEST--
NDArray::count() returns 0 for 0-D scalar arrays (no axis 0 to enumerate)
--FILE--
<?php
/* 0-D arrays have an empty shape (). PHP's count() on the empty-array form
   shape() returns is 0, so count() on the NDArray must agree. This also
   guards against a regression where count() read uninitialised memory from
   the dimensions buffer of 0-D scalars. */

/* Scalar from int. */
$si = new NDArray(42);
echo $si->count() . PHP_EOL;
echo count($si) . PHP_EOL;

/* Scalar from float. */
$sf = new NDArray(3.14);
echo $sf->count() . PHP_EOL;
echo count($sf) . PHP_EOL;

/* Scalar with explicit dtype. */
$sd = new NDArray(7, 'int32');
echo $sd->count() . PHP_EOL;
echo count($sd) . PHP_EOL;

/* shape() should also return an empty array on a 0-D NDArray -- that and
   count() == 0 together describe a consistent "no axis" state. */
print_r($si->shape());

/* The total element count is still 1 (size() != count() for 0-D). */
echo $si->size() . PHP_EOL;

/* Repeated calls must not mutate the array or its shape. */
for ($i = 0; $i < 10; $i++) {
    if ($si->count() !== 0) {
        echo "regression at iter $i\n";
        exit(1);
    }
}
echo "stable\n";
?>
--EXPECT--
0
0
0
0
0
0
Array
(
)
1
stable
