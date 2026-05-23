--TEST--
NumPower::array sparse intake survives a CPU→GPU transfer and on-device arithmetic
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* The fix is in the CPU-side factory, but the resulting NDArray is regularly
 * transferred to GPU and used in cuBLAS / kernel paths. Exercise that
 * pipeline so a regression in the intake also surfaces as a GPU failure. */

$x = [1, 2, 3, 4, 5];
unset($x[0]);            /* keys 1..4, values 2..5 */

$dtypes = ['float32', 'float64', 'float16', 'int8', 'uint8',
           'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float128'];

foreach ($dtypes as $dt) {
    $a = NumPower::array([1, 2, 3, 4], $dt)->gpu();
    $b = NumPower::array($x,            $dt)->gpu();
    $sum = $a + $b;                                          /* on-device add */
    echo $dt, ': ', $sum, "\n";
}

/* 2-D sparse outer on GPU */
$m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
unset($m[0]);
$g = NumPower::array($m, 'float32')->gpu();
$result = $g + $g;
echo "2d_outer_sparse_gpu:\n", $result, "\n";
echo "2d_outer_sparse_gpu shape: ", json_encode($result->shape()), "\n";
?>
--EXPECT--
float32: [3, 5, 7, 9]
float64: [3, 5, 7, 9]
float16: [3, 5, 7, 9]
int8: [3, 5, 7, 9]
uint8: [3, 5, 7, 9]
int16: [3, 5, 7, 9]
uint16: [3, 5, 7, 9]
int32: [3, 5, 7, 9]
uint32: [3, 5, 7, 9]
int64: [3, 5, 7, 9]
uint64: [3, 5, 7, 9]
float128: [3, 5, 7, 9]
2d_outer_sparse_gpu:
[[8, 10, 12]
 [14, 16, 18]]
2d_outer_sparse_gpu shape: [2,3]
