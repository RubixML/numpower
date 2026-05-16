--TEST--
NumPower::array with explicit float64 dtype
--FILE--
<?php
$a = NumPower::array([1, 2, 3, 4], 'float64');
echo $a . "\n";
$b = NumPower::array([[1, 2], [3, 4]], 'float64');
echo $b . "\n";
?>
--EXPECT--
[1, 2, 3, 4]
[[1, 2]
 [3, 4]]
