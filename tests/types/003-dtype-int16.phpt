--TEST--
NDArray int16 dtype: creation, display, boundary values
--FILE--
<?php
$a = new NDArray([0, 1000, -32768, 32767], "int16");
echo $a;
?>
--EXPECT--
[0, 1000, -32768, 32767]
