--TEST--
NumPower::array with float32 dtype
--FILE--
<?php
$a = NumPower::array([1, 2, 3, 4], 'float32');
echo $a . "\n";
$b = NumPower::array([[1, 2], [3, 4]], 'float32');
echo $b . "\n";
?>
--EXPECT--
[1, 2, 3, 4]
[[1, 2]
 [3, 4]]
