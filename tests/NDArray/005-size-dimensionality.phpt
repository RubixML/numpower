--TEST--
NDArray::size() returns correct total element count for arrays of different dimensionality
--FILE--
<?php
// 0-D scalar
$scalar = new NDArray(5);
echo $scalar->size() . PHP_EOL;

// 1-D (3,)
$v = new NDArray([1, 2, 3]);
echo $v->size() . PHP_EOL;

// 2-D non-square (2 rows, 3 cols) → 6 elements
$m = NumPower::array([[1, 2, 3], [4, 5, 6]]);
echo $m->size() . PHP_EOL;

// 3-D (2×3×4) → 24 elements
$t3 = NumPower::zeros([2, 3, 4]);
echo $t3->size() . PHP_EOL;

// 4-D (2×3×4×5) → 120 elements
$t4 = NumPower::zeros([2, 3, 4, 5]);
echo $t4->size() . PHP_EOL;

// single-element 2-D (1×1) → 1
$one = NumPower::ones([1, 1]);
echo $one->size() . PHP_EOL;

// column vector (5×1) → 5
$col = NumPower::zeros([5, 1]);
echo $col->size() . PHP_EOL;

// row vector (1×5) → 5
$row = NumPower::zeros([1, 5]);
echo $row->size() . PHP_EOL;
?>
--EXPECT--
1
3
6
24
120
1
5
5
