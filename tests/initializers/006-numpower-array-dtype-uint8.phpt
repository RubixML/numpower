--TEST--
NumPower::array with uint8 dtype: boundary values
--FILE--
<?php
$a = NumPower::array([0, 1, 127, 255], 'uint8');
echo $a . "\n";
?>
--EXPECT--
[0, 1, 127, 255]
