--TEST--
NDArray::count() reflects the leading axis after reshape -- not the total size
--FILE--
<?php
/* count() must return shape[0] of the CURRENT shape, regardless of the
   underlying buffer size. This differs from size(), which tracks the
   product of all dimensions. */

$flat = NumPower::arange(24.0);
echo $flat->count() . PHP_EOL;          /* 24 - 1-D shape (24,) */

$mat = NumPower::reshape($flat, [4, 6]);
echo $mat->count() . PHP_EOL;           /* 4  - 2-D shape (4,6) */

$cube = NumPower::reshape($flat, [2, 3, 4]);
echo $cube->count() . PHP_EOL;          /* 2  - 3-D shape (2,3,4) */

$t4 = NumPower::reshape($flat, [2, 2, 2, 3]);
echo $t4->count() . PHP_EOL;            /* 2  - 4-D shape (2,2,2,3) */

$row = NumPower::reshape($flat, [1, 24]);
echo $row->count() . PHP_EOL;           /* 1  - 2-D shape (1,24) */

$col = NumPower::reshape($flat, [24, 1]);
echo $col->count() . PHP_EOL;           /* 24 - 2-D shape (24,1) */

/* size() vs count() invariants: size never changes under reshape; count does. */
echo $mat->size() . ':' . $mat->count() . PHP_EOL;   /* 24:4 */
echo $cube->size() . ':' . $cube->count() . PHP_EOL; /* 24:2 */
echo $row->size() . ':' . $row->count() . PHP_EOL;   /* 24:1 */
?>
--EXPECT--
24
4
2
2
1
24
24:4
24:2
24:1
