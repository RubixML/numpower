--TEST--
NumPower::prod
--FILE--
<?php
$a = NumPower::array([[-156.50, 150.525435], [1, -39.151414]]);
print_r(NumPower::prod($a));
print_r(NumPower::prod($a, axis: 0)->toArray());
print_r(NumPower::prod($a, axis: 1)->toArray());
print_r(NumPower::prod($a[0]));
print_r(NumPower::prod([[0.12],[-0.513124]]));
?>
--EXPECT--
922298.87121119Array
(
    [0] => -156.5
    [1] => -5893.2836914062
)
Array
(
    [0] => -23557.23046875
    [1] => -39.151412963867
)
-23557.230796814-0.061574877316321