--TEST--
serialize() / unserialize() round-trip preserves 2-D and 3-D structure
--FILE--
<?php
/* Validate that the structured serialize/unserialize round-trip preserves
   the array shape (not just the flattened element values). */

/* 2-D */
$cases2d = [
    'float32'  => [[1.5, 2.5], [3.5, 4.5]],
    'float128' => [['1.5','2.5'],['3.5','4.5']],
    'int64'    => [[PHP_INT_MIN, 0], [1, PHP_INT_MAX]],
    'uint64'   => [['0','1'],['2','18446744073709551615']],
];
foreach ($cases2d as $t => $d) {
    $a = new NDArray($d, $t);
    $b = unserialize(serialize($a));
    echo "$t 2D: ", ($a->toArray() === $b->toArray() ? "OK" : "BAD"), "\n";
}

/* 3-D */
$cases3d = [
    'float64'  => [[[1.5,2.5],[3.5,4.5]],[[5.5,6.5],[7.5,8.5]]],
    'float128' => [[['1','2'],['3','4']],[['5','6'],['7','8']]],
    'uint64'   => [[['0','1'],['2','3']],[['4','5'],['6','18446744073709551615']]],
];
foreach ($cases3d as $t => $d) {
    $a = new NDArray($d, $t);
    $b = unserialize(serialize($a));
    echo "$t 3D: ", ($a->toArray() === $b->toArray() ? "OK" : "BAD"), "\n";
}
?>
--EXPECT--
float32 2D: OK
float128 2D: OK
int64 2D: OK
uint64 2D: OK
float64 3D: OK
float128 3D: OK
uint64 3D: OK
