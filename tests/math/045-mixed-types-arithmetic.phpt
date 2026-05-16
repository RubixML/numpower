--TEST--
Arithmetic type promotion: float16 op float64 promotes to float64; float16 op float16 stays float16
--FILE--
<?php
$a = new NDArray([[1,2],[3,4]], 'float16');
$b = new NDArray([[3,4],[5,6]], 'float64');

echo "float16 + float64:\n";
print_r(($a + $b)->toArray());

echo "float16 - float64:\n";
print_r(($a - $b)->toArray());

echo "float16 * float64:\n";
print_r(($a * $b)->toArray());

echo "float16 / float64:\n";
print_r(($a / $b)->toArray());

$c = new NDArray([[1,2],[3,4]], 'float16');
$d = new NDArray([[1,2],[3,4]], 'float16');

echo "float16 + float16:\n";
print_r(($c + $d)->toArray());

echo "float16 * float16:\n";
print_r(($c * $d)->toArray());
?>
--EXPECT--
float16 + float64:
Array
(
    [0] => Array
        (
            [0] => 4
            [1] => 6
        )

    [1] => Array
        (
            [0] => 8
            [1] => 10
        )

)
float16 - float64:
Array
(
    [0] => Array
        (
            [0] => -2
            [1] => -2
        )

    [1] => Array
        (
            [0] => -2
            [1] => -2
        )

)
float16 * float64:
Array
(
    [0] => Array
        (
            [0] => 3
            [1] => 8
        )

    [1] => Array
        (
            [0] => 15
            [1] => 24
        )

)
float16 / float64:
Array
(
    [0] => Array
        (
            [0] => 0.33333333333333
            [1] => 0.5
        )

    [1] => Array
        (
            [0] => 0.6
            [1] => 0.66666666666667
        )

)
float16 + float16:
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
        )

    [1] => Array
        (
            [0] => 6
            [1] => 8
        )

)
float16 * float16:
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 4
        )

    [1] => Array
        (
            [0] => 9
            [1] => 16
        )

)
