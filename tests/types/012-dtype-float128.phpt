--TEST--
NDArray float128 dtype: creation from strings, high-precision display
--FILE--
<?php
$a = new NDArray(["1.23456789012345678901234"], "float128");
echo $a;
$b = new NDArray(["3.14159265358979323846264338327950288"], "float128");
echo $b;
?>
--EXPECT--
[1.23456789012345679]
[3.14159265358979324]
