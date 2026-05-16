--TEST--
GPU arithmetic with mixed types promotes to the higher type (PyTorch semantics)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
// float16 GPU + float64 GPU => float64 result on GPU
$a = new NDArray([[1,2],[3,4]], 'float16');
$b = NumPower::array([[3,4],[5,6]]);  // float64

$ag = $a->gpu();
$bg = $b->gpu();

echo "float16+float64 GPU add:\n";
print_r(($ag + $bg)->cpu()->toArray());

echo "float16+float64 GPU sub:\n";
print_r(($ag - $bg)->cpu()->toArray());

echo "float16+float64 GPU mul:\n";
print_r(($ag * $bg)->cpu()->toArray());

// float16 GPU + float16 GPU => float16 result (via float32 compute)
$c  = new NDArray([[1,2],[3,4]], 'float16');
$d  = new NDArray([[1,2],[3,4]], 'float16');
$cg = $c->gpu();
$dg = $d->gpu();

echo "float16+float16 GPU add:\n";
print_r(($cg + $dg)->cpu()->toArray());
?>
--EXPECT--
float16+float64 GPU add:
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
float16+float64 GPU sub:
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
float16+float64 GPU mul:
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
float16+float16 GPU add:
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
