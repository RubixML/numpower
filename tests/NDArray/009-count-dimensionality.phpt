--TEST--
NDArray::count() returns the size of axis 0 for arrays of any dimensionality
--FILE--
<?php
/* 0-D scalar (shape []): no axis 0 to enumerate -> 0. */
$scalar = new NDArray(5);
echo $scalar->count() . PHP_EOL;

/* 1-D (3,) -> 3. */
$v = new NDArray([1, 2, 3]);
echo $v->count() . PHP_EOL;

/* 2-D non-square (2 rows, 3 cols) -> 2. */
$m = NumPower::array([[1, 2, 3], [4, 5, 6]]);
echo $m->count() . PHP_EOL;

/* 3-D (2x3x4) -> 2. */
$t3 = NumPower::zeros([2, 3, 4]);
echo $t3->count() . PHP_EOL;

/* 4-D (7x3x4x5) -> 7. */
$t4 = NumPower::zeros([7, 3, 4, 5]);
echo $t4->count() . PHP_EOL;

/* single-element 2-D (1x1) -> 1. */
$one = NumPower::ones([1, 1]);
echo $one->count() . PHP_EOL;

/* column vector (5x1) -> 5. */
$col = NumPower::zeros([5, 1]);
echo $col->count() . PHP_EOL;

/* row vector (1x5) -> 1. */
$row = NumPower::zeros([1, 5]);
echo $row->count() . PHP_EOL;

/* large leading axis -> exact value. */
$big = NumPower::zeros([1024, 8]);
echo $big->count() . PHP_EOL;
?>
--EXPECT--
0
3
2
2
7
1
5
1
1024
