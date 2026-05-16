--TEST--
GPU arithmetic operations on float32 arrays
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
$a  = new NDArray([[1,2],[3,4]], 'float32');
$b  = new NDArray([[3,4],[5,6]], 'float32');
$ag = $a->gpu();
$bg = $b->gpu();

print_r(($ag + $bg)->cpu()->toArray());
print_r(($ag - $bg)->cpu()->toArray());
print_r(($ag * $bg)->cpu()->toArray());
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
            [0] => 3
            [1] => 8
        )

    [1] => Array
        (
            [0] => 15
            [1] => 24
        )

)
