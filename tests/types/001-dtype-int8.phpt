--TEST--
NDArray int8 dtype: creation, display, boundary values
--FILE--
<?php
$a = new NDArray([0, 1, -1, 127, -128], "int8");
echo $a;
?>
--EXPECT--
[0, 1, -1, 127, -128]
