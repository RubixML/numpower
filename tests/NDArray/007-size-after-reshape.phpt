--TEST--
NDArray::size() remains invariant under NumPower::reshape()
--FILE--
<?php
$flat = NumPower::arange(24.0);

// original 1-D → 24 elements
echo $flat->size() . PHP_EOL;

// 1-D (24,) → 2-D (4×6)
$mat = NumPower::reshape($flat, [4, 6]);
echo $mat->size() . PHP_EOL;

// 1-D (24,) → 3-D (2×3×4)
$cube = NumPower::reshape($flat, [2, 3, 4]);
echo $cube->size() . PHP_EOL;

// 1-D (24,) → 4-D (2×2×2×3)
$t4 = NumPower::reshape($flat, [2, 2, 2, 3]);
echo $t4->size() . PHP_EOL;

// 1-D (24,) → 2-D (1×24) — degenerate row
$row = NumPower::reshape($flat, [1, 24]);
echo $row->size() . PHP_EOL;
?>
--EXPECT--
24
24
24
24
24
