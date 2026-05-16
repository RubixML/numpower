--TEST--
NDArray float4 E2M1 dtype: creation from strings, all 8 positive representable values
--FILE--
<?php
// All positive float4 E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
$a = new NDArray(["0.0", "0.5", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0"], "float4");
echo $a;
// Negative values
$b = new NDArray(["-0.5", "-1.0", "-2.0", "-6.0"], "float4");
echo $b;
?>
--EXPECT--
[0, 0.5, 1, 1.5, 2, 3, 4, 6]
[-0.5, -1, -2, -6]
