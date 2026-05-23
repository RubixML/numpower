--TEST--
NumPower::sum
--FILE--
<?php
$a = NumPower::array([[-156.50, 150.525435], [0, -39.151414]]);
print_r(NumPower::sum($a));
print_r(NumPower::sum($a, axis: 0)->toArray());
print_r(NumPower::sum($a, axis: 1)->toArray());
print_r(NumPower::sum($a[0]));
print_r(NumPower::sum([[0.12],[-0.513124]]));
?>
--EXPECT--
-45.1259765625Array
(
    [0] => -156.5
    [1] => 111.3740234375
)
Array
(
    [0] => -5.9745635986328
    [1] => -39.151412963867
)
-5.9745635986328-0.39312399178743