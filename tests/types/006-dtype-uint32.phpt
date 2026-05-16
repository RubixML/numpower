--TEST--
NDArray uint32 dtype: creation, display, boundary values
--FILE--
<?php
$a = new NDArray([0, 1, 2147483648, 4294967295], "uint32");
echo $a;
?>
--EXPECT--
[0, 1, 2147483648, 4294967295]
