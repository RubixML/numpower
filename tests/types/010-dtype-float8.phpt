--TEST--
NDArray float8 E4M3 dtype: creation from strings, display
--FILE--
<?php
$a = new NDArray(["0.0", "1.0", "-2.0", "240.0"], "float8");
echo $a;
?>
--EXPECT--
[0, 1, -2, 240]
