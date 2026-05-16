--TEST--
NDArray::shape() returns correct shape for arrays residing in GPU VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
// 1-D float32 on GPU
$a = NumPower::array([1, 2, 3, 4, 5], 'float32')->gpu();
print_r($a->shape());

// 2-D float64 on GPU
$b = NumPower::array([[1, 2, 3], [4, 5, 6]], 'float64')->gpu();
print_r($b->shape());

// 2-D int32 on GPU
$c = NumPower::array([[1, 2], [3, 4]], 'int32')->gpu();
print_r($c->shape());

// 3-D float64 on GPU (via zeros)
$d = NumPower::zeros([2, 3, 4])->gpu();
print_r($d->shape());
?>
--EXPECT--
Array
(
    [0] => 5
)
Array
(
    [0] => 2
    [1] => 3
)
Array
(
    [0] => 2
    [1] => 2
)
Array
(
    [0] => 2
    [1] => 3
    [2] => 4
)
