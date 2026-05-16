--TEST--
NumPower::array with NDArray input returns the same object
--FILE--
<?php
$a = NumPower::array([1, 2, 3], 'int32');
$b = NumPower::array($a);
echo $b . "\n";
var_dump($a === $b);
?>
--EXPECT--
[1, 2, 3]
bool(true)
