--TEST--
NumPower::expm1
--FILE--
<?php
$a = NumPower::array([[1, 2], [3, 4]]);
print_r(NumPower::expm1($a)->toArray());
print_r(NumPower::expm1($a[0])->toArray());
print_r(NumPower::expm1([[1],[2]])->toArray());
?>
--EXPECTF--
Array
(
    [0] => Array
        (
            [0] => %f
            [1] => 6.3890562057495
        )

    [1] => Array
        (
            [0] => 19.085536956787
            [1] => %f
        )

)
Array
(
    [0] => %f
    [1] => 6.3890562057495
)
Array
(
    [0] => Array
        (
            [0] => %f
        )

    [1] => Array
        (
            [0] => 6.3890562057495
        )

)