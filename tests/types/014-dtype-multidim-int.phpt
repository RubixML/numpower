--TEST--
NDArray integer dtypes: 2D arrays
--FILE--
<?php
$a = new NDArray([[1, 2, 3], [4, 5, 6]], "int32");
echo $a . "\n";
$b = new NDArray([[0, 255], [128, 64]], "uint8");
echo $b . "\n";
?>
--EXPECT--
[[1, 2, 3]
 [4, 5, 6]]
[[0, 255]
 [128, 64]]
