--TEST--
NDArray::count() returns correct axis-0 length for arrays residing in GPU VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* count() reads metadata only -- the value must be identical on CPU and GPU
   and must not trigger a device->host transfer. */

/* 1-D float32 on GPU. */
$a = NumPower::array([1, 2, 3, 4, 5], 'float32')->gpu();
echo $a->count() . PHP_EOL;

/* 2-D float64 on GPU (2x3). */
$b = NumPower::array([[1, 2, 3], [4, 5, 6]], 'float64')->gpu();
echo $b->count() . PHP_EOL;

/* 2-D int32 on GPU (4x2). */
$c = NumPower::array([[1, 2], [3, 4], [5, 6], [7, 8]], 'int32')->gpu();
echo $c->count() . PHP_EOL;

/* 3-D float64 on GPU (2x3x4) via zeros. */
$d = NumPower::zeros([2, 3, 4])->gpu();
echo $d->count() . PHP_EOL;

/* 4-D float64 on GPU (7x3x4x5). */
$e = NumPower::zeros([7, 3, 4, 5])->gpu();
echo $e->count() . PHP_EOL;

/* Large leading axis on GPU. */
$big = NumPower::zeros([1024, 8])->gpu();
echo $big->count() . PHP_EOL;

/* CPU<->GPU parity for every supported dtype. count() reads only metadata,
   so it must be identical on CPU and GPU regardless of dtype -- including
   the emulated ones (float4, float8, float128) and uint64. */
$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];
foreach ($dtypes as $t) {
    $cpu = NumPower::array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], $t);
    $gpu = $cpu->gpu();
    echo $t . ':' . $cpu->count() . '=' . $gpu->count() . PHP_EOL;
}
?>
--EXPECT--
5
2
4
2
7
1024
float4:3=3
float8:3=3
float16:3=3
float32:3=3
float64:3=3
float128:3=3
int8:3=3
uint8:3=3
int16:3=3
uint16:3=3
int32:3=3
uint32:3=3
int64:3=3
uint64:3=3
