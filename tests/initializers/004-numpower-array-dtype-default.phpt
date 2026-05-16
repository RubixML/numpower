--TEST--
NumPower::array without dtype uses float64 by default
--FILE--
<?php
$a = NumPower::array([1, 2, 3]);
echo $a . "\n";
$b = NumPower::array([[1, 2], [3, 4]]);
echo $b . "\n";
?>
--EXPECT--
[1, 2, 3]
[[1, 2]
 [3, 4]]
