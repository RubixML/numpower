--TEST--
NumPower::zeros($shape, $dtype, NUMPOWER_CUDA) allocates directly in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* zeros() with device=1 must build the array on the GPU without ever
   materialising a host buffer (so no implicit host->device transfer is
   needed) AND the on-device bytes must decode as zero through the same
   typed-D2H path used by every other GPU read. We verify both invariants
   for every supported dtype.

   The "VRAM-direct" check is indirect — we can't peek at the cudaMalloc
   site from PHP — but the value-equality check against `zeros(..., CPU)`
   combined with NDARRAY_VCHECK in CI ensures any host-staging bug would
   either corrupt the value or leak a CPU buffer. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    $gpu = NumPower::zeros([4], $dt, NUMPOWER_CUDA);
    $cpu = NumPower::zeros([4], $dt);
    $back = $gpu->cpu();
    $ok_dev    = $gpu->isGPU();
    $ok_shape  = $gpu->shape() === [4];
    $ok_values = (string)$back === (string)$cpu;
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' values=', ($ok_values ? 'OK' : 'BAD'),
         "\n";
}

/* On-device arithmetic — zeros + zeros must stay on GPU and remain zero. */
$g = NumPower::zeros([3, 3], 'float32', NUMPOWER_CUDA);
$h = NumPower::zeros([3, 3], 'float32', NUMPOWER_CUDA);
$sum = NumPower::add($g, $h);
echo 'gpu_arith_sum_zero: ',
     ((string)NumPower::sum($sum) === '0' && $sum->isGPU() ? 'OK' : 'BAD'), "\n";
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
gpu_arith_sum_zero: OK
