--TEST--
NumPower::array with dtype on 2D arrays
--FILE--
<?php
$a = NumPower::array([[1, 2, 3], [4, 5, 6]], 'int32');
echo $a . "\n";
$b = NumPower::array([[0, 255], [128, 64]], 'uint8');
echo $b . "\n";
?>
--EXPECT--
[[1, 2, 3]
 [4, 5, 6]]
[[0, 255]
 [128, 64]]
