--TEST--
NumPower::log10
--FILE--
<?php
$a = NumPower::array([[1, 2], [3, 4]]);
print_r(NumPower::log10($a)->toArray());
print_r(NumPower::log10($a[0])->toArray());
print_r(NumPower::log10([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0.30103000998497
        )

    [1] => Array
        (
            [0] => 0.47712126374245
            [1] => 0.60206001996994
        )

)
Array
(
    [0] => 0
    [1] => 0.30103000998497
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 0.30103000998497
        )

)