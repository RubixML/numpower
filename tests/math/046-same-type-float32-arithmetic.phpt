--TEST--
Arithmetic operations between same-type float32 arrays (CPU)
--FILE--
<?php
$a = new NDArray([[1,2],[3,4]], 'float32');
$b = new NDArray([[3,4],[5,6]], 'float32');

print_r(($a + $b)->toArray());
print_r(($a * $b)->toArray());
print_r(($a - $b)->toArray());
print_r(($a / $b)->toArray());
?>
--EXPECT--
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
Array
(
    [0] => Array
        (
            [0] => 0.33333334326744
            [1] => 0.5
        )

    [1] => Array
        (
            [0] => 0.60000002384186
            [1] => 0.66666668653488
        )

)
