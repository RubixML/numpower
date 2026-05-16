--TEST--
NumPower::array with int32 dtype: boundary values
--FILE--
<?php
$a = NumPower::array([0, 1, -2147483648, 2147483647], 'int32');
echo $a . "\n";
?>
--EXPECT--
[0, 1, -2147483648, 2147483647]
