--TEST--
Mixed-dtype promotion follows PyTorch: float wins over int, wider type wins within a category, int+uint of same width promotes to next signed
--FILE--
<?php
function show($r) {
    $v = $r[0];
    $t = is_int($v) ? 'int' : (is_float($v) ? 'float' : (is_string($v) ? 'string' : gettype($v)));
    return "$t/" . (string)$v;
}

/* float vs int → float */
$a = new NDArray([1, 1, 1, 1], 'int32');
$b = new NDArray([2.0, 2.0, 2.0, 2.0], 'float64');
echo "int32 + float64 = ", show($a + $b), "\n";

/* same-category wider wins */
$a = new NDArray([1, 1, 1, 1], 'int8');
$b = new NDArray([2, 2, 2, 2], 'int32');
echo "int8 + int32 = ", show($a + $b), "\n";

/* int + uint same width → next signed */
$a = new NDArray([1, 1, 1, 1], 'int8');
$b = new NDArray([2, 2, 2, 2], 'uint8');
echo "int8 + uint8 = ", show($a + $b), "\n";

/* float16 + float64 → float64 */
$a = new NDArray([1.0, 1.0, 1.0, 1.0], 'float16');
$b = new NDArray([2.0, 2.0, 2.0, 2.0], 'float64');
echo "float16 + float64 = ", show($a + $b), "\n";

/* Same dtype both operands preserves dtype */
$a = new NDArray([5, 5, 5, 5], 'int8');
$b = new NDArray([5, 5, 5, 5], 'int8');
echo "int8 + int8 = ", show($a + $b), "\n";

/* Float scalar with int tensor → float */
$a = new NDArray([1, 2, 3, 4], 'int32');
echo "int32 + 0.5 = ", show($a + 0.5), "\n";

/* Int scalar with int tensor → preserves int dtype */
$a = new NDArray([1, 2, 3, 4], 'int32');
echo "int32 + 1 = ", show($a + 1), "\n";
?>
--EXPECT--
int32 + float64 = float/3
int8 + int32 = int/3
int8 + uint8 = int/3
float16 + float64 = float/3
int8 + int8 = int/10
int32 + 0.5 = float/1.5
int32 + 1 = int/2
