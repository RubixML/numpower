--TEST--
NDArray::shape() reflects the new shape after NumPower::reshape()
--FILE--
<?php
$flat = NumPower::arange(12.0);

// 1-D (12,) → 2-D (3×4)
$mat = NumPower::reshape($flat, [3, 4]);
print_r($mat->shape());

// 1-D (12,) → 3-D (2×2×3)
$cube = NumPower::reshape($flat, [2, 2, 3]);
print_r($cube->shape());
?>
--EXPECT--
Array
(
    [0] => 3
    [1] => 4
)
Array
(
    [0] => 2
    [1] => 2
    [2] => 3
)
