--TEST--
NDArray float128 dtype: toArray returns strings preserving precision
--FILE--
<?php
$a = new NDArray(["1.23456789012345678901234", "3.14159265358979323846264338327950288"], "float128");
$arr = $a->toArray();

// Elements must be strings (PHP float/double cannot represent float128 precision)
var_dump(gettype($arr[0]) === 'string');
var_dump(gettype($arr[1]) === 'string');

// Values should preserve at least 18 significant digits
echo $arr[0] . "\n";
echo $arr[1] . "\n";
?>
--EXPECT--
bool(true)
bool(true)
1.23456789012345679
3.14159265358979324
