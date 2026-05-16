--TEST--
NDArray int64 dtype: creation, display, PHP_INT_MAX/MIN boundary values
--FILE--
<?php
$a = new NDArray([0, 1, -1, PHP_INT_MAX, PHP_INT_MIN], "int64");
echo $a;
?>
--EXPECT--
[0, 1, -1, 9223372036854775807, -9223372036854775808]
