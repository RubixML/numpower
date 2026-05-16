--TEST--
NDArray int32 dtype: creation, display, boundary values
--FILE--
<?php
$a = new NDArray([0, 1, -2147483648, 2147483647], "int32");
echo $a;
?>
--EXPECT--
[0, 1, -2147483648, 2147483647]
