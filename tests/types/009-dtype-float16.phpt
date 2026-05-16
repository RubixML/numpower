--TEST--
NDArray float16 dtype: creation from strings, display, representable values
--FILE--
<?php
$a = new NDArray(["0.5", "-1.5", "2.0", "65504.0"], "float16");
echo $a . "\n";
$b = new NDArray([["0.5", "1.0"], ["-1.5", "2.0"]], "float16");
echo $b . "\n";
?>
--EXPECT--
[0.5, -1.5, 2, 65504]
[[0.5, 1]
 [-1.5, 2]]
