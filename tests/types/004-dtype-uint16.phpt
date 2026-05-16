--TEST--
NDArray uint16 dtype: creation, display, boundary values
--FILE--
<?php
$a = new NDArray([0, 1000, 32768, 65535], "uint16");
echo $a;
?>
--EXPECT--
[0, 1000, 32768, 65535]
