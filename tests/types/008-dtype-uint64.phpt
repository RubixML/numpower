--TEST--
NDArray uint64 dtype: creation from strings, display, large values
--FILE--
<?php
// uint64 max exceeds PHP_INT_MAX, so large values must come as strings
$a = new NDArray(["0", "1", "9223372036854775808", "18446744073709551615"], "uint64");
echo $a . "\n";
// small values can also be PHP ints
$b = new NDArray([0, 100, 65535], "uint64");
echo $b . "\n";
?>
--EXPECT--
[0, 1, 9223372036854775808, 18446744073709551615]
[0, 100, 65535]
