--TEST--
NDArray uint8 dtype: creation, display, boundary values
--FILE--
<?php
$a = new NDArray([0, 1, 128, 255], "uint8");
echo $a;
?>
--EXPECT--
[0, 1, 128, 255]
