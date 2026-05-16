--TEST--
NumPower::sqrt
--FILE--
<?php
$a = NumPower::array([[-156, 150], [19, -39]]);
print_r(NumPower::sqrt($a)->toArray());
print_r(NumPower::sqrt($a[0])->toArray());
print_r(NumPower::sqrt([[0],[-0.5]])->toArray());
?>
--EXPECTF--
%A
Array
(
    [0] => Array
        (
            [0] => NAN
            [1] => 12.247448921204
        )

    [1] => Array
        (
            [0] => 4.3588991165161
            [1] => NAN
        )

)
%A
Array
(
    [0] => NAN
    [1] => 12.247448921204
)
%A
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => NAN
        )

)