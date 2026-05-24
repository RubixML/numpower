--TEST--
NumPower::ones($shape, $dtype, NUMPOWER_CUDA) allocates directly in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* ones() with device=1 must build the array on the GPU without ever
   materialising the full result on the host (only a single elsize-byte
   seed crosses the bus — see cuda_fill_bytes) AND the on-device bytes
   must decode as 1 through the same typed-D2H path used by every other
   GPU read. Verify both invariants for every supported dtype. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    $gpu = NumPower::ones([4], $dt, NUMPOWER_CUDA);
    $cpu = NumPower::ones([4], $dt);
    $back = $gpu->cpu();
    $ok_dev    = $gpu->isGPU();
    $ok_shape  = $gpu->shape() === [4];
    $ok_values = (string)$back === (string)$cpu;
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' values=', ($ok_values ? 'OK' : 'BAD'),
         "\n";
}

/* On-device arithmetic — ones + ones must stay on GPU and produce 2s. */
$g = NumPower::ones([3, 3], 'float32', NUMPOWER_CUDA);
$h = NumPower::ones([3, 3], 'float32', NUMPOWER_CUDA);
$sum = NumPower::add($g, $h);
echo 'gpu_arith_sum_eq_2N: ',
     ((string)NumPower::sum($sum) === '18' && $sum->isGPU() ? 'OK' : 'BAD'), "\n";

/* Larger boundary-aligned sweep — covers the doubling broadcast at sizes
   that exercise odd-tail copy paths in cuda_fill_bytes. */
foreach ([1, 7, 8, 9, 1024, 1025] as $n) {
    $a = NumPower::ones([$n], 'float64', NUMPOWER_CUDA);
    $ok = ((string)NumPower::sum($a) === (string)$n);
    echo 'gpu_n=', $n, ': ', ($ok ? 'OK' : 'BAD'), "\n";
}
?>
--EXPECT--
float4: gpu=1 shape=OK values=OK
float8: gpu=1 shape=OK values=OK
float16: gpu=1 shape=OK values=OK
float32: gpu=1 shape=OK values=OK
float64: gpu=1 shape=OK values=OK
float128: gpu=1 shape=OK values=OK
int8: gpu=1 shape=OK values=OK
uint8: gpu=1 shape=OK values=OK
int16: gpu=1 shape=OK values=OK
uint16: gpu=1 shape=OK values=OK
int32: gpu=1 shape=OK values=OK
uint32: gpu=1 shape=OK values=OK
int64: gpu=1 shape=OK values=OK
uint64: gpu=1 shape=OK values=OK
gpu_arith_sum_eq_2N: OK
gpu_n=1: OK
gpu_n=7: OK
gpu_n=8: OK
gpu_n=9: OK
gpu_n=1024: OK
gpu_n=1025: OK
