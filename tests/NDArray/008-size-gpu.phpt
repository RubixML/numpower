--TEST--
NDArray::size() returns correct element count for arrays residing in GPU VRAM
--FILE--
<?php
// 1-D float32 on GPU
$a = NumPower::array([1, 2, 3, 4, 5], 'float32')->gpu();
echo $a->size() . PHP_EOL;

// 2-D float64 on GPU (2×3)
$b = NumPower::array([[1, 2, 3], [4, 5, 6]], 'float64')->gpu();
echo $b->size() . PHP_EOL;

// 2-D int32 on GPU (2×2)
$c = NumPower::array([[1, 2], [3, 4]], 'int32')->gpu();
echo $c->size() . PHP_EOL;

// 3-D float64 on GPU (2×3×4) via zeros
$d = NumPower::zeros([2, 3, 4])->gpu();
echo $d->size() . PHP_EOL;

// 4-D float64 on GPU (2×3×4×5)
$e = NumPower::zeros([2, 3, 4, 5])->gpu();
echo $e->size() . PHP_EOL;
?>
--EXPECT--
5
6
4
24
120
