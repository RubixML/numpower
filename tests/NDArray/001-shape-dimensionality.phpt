--TEST--
NDArray::shape() returns correct shape for arrays of different dimensionality
--FILE--
<?php
// 0-D scalar
$scalar = new NDArray(5);
print_r($scalar->shape());

// 1-D
$v = new NDArray([1, 2, 3]);
print_r($v->shape());

// 2-D non-square (2 rows, 3 cols)
$m = NumPower::array([[1, 2, 3], [4, 5, 6]]);
print_r($m->shape());

// 3-D
$t3 = NumPower::zeros([2, 3, 4]);
print_r($t3->shape());

// 4-D
$t4 = NumPower::zeros([2, 3, 4, 5]);
print_r($t4->shape());
?>
--EXPECT--
Array
(
)
Array
(
    [0] => 3
)
Array
(
    [0] => 2
    [1] => 3
)
Array
(
    [0] => 2
    [1] => 3
    [2] => 4
)
Array
(
    [0] => 2
    [1] => 3
    [2] => 4
    [3] => 5
)
