--TEST--
NDArray Iterator: works on arrays produced by every NumPower factory (zeros / ones / arange / identity / full / array)
--FILE--
<?php
/* Every NumPower factory path eventually calls into the NDArray constructor
   pipeline that installs both iterator and php_iterator. Iterate one product
   from each factory to confirm none of them skip iterator init. */

$tests = [
    'zeros'      => NumPower::zeros([3, 2]),
    'ones'       => NumPower::ones([3, 2]),
    'arange'     => NumPower::arange(6),
    'identity'   => NumPower::identity(3),
    'full'       => NumPower::full([3, 2], 7),
    'array1d'    => NumPower::array([10, 20, 30], 'int32'),
    'array2d'    => NumPower::array([[1, 2], [3, 4]], 'float64'),
    'fromInt'    => NumPower::array([1, 2, 3]),
    'standardNorm' => NumPower::standardNormal([4]),  /* random but iterates */
];

foreach ($tests as $name => $arr) {
    $count = 0;
    foreach ($arr as $k => $v) {
        $count++;
        if ($count > 100) break;  /* safety belt */
    }
    echo "$name: count=", $arr->count(),
         " foreach_steps=", $count,
         " match=", ($count === $arr->count()) ? 'OK' : 'BAD', "\n";
}

/* Also exercise the explicit driver on a single factory result. */
$a = NumPower::arange(5);
$a->rewind();
$seq = [];
while ($a->valid()) {
    $seq[] = (string)$a->current();
    $a->next();
}
echo "arange seq: ", implode(',', $seq), "\n";
?>
--EXPECT--
zeros: count=3 foreach_steps=3 match=OK
ones: count=3 foreach_steps=3 match=OK
arange: count=6 foreach_steps=6 match=OK
identity: count=3 foreach_steps=3 match=OK
full: count=3 foreach_steps=3 match=OK
array1d: count=3 foreach_steps=3 match=OK
array2d: count=2 foreach_steps=2 match=OK
fromInt: count=3 foreach_steps=3 match=OK
standardNorm: count=4 foreach_steps=4 match=OK
arange seq: 0,1,2,3,4
