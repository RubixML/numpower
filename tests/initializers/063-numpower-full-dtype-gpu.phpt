--TEST--
NumPower::full($shape, $value, $dtype, NUMPOWER_CUDA) allocates directly in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* full() with device=1 must build the array on the GPU without ever
   materialising the full result on the host — only a single elsize-byte
   seed crosses the bus, via cuda_fill_bytes. The on-device bytes must
   decode through TypedD2H to the same value the CPU path produces. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    $gpu = NumPower::full([4], 2, $dt, NUMPOWER_CUDA);
    $cpu = NumPower::full([4], 2, $dt, NUMPOWER_CPU);
    $back = $gpu->cpu();
    $ok_dev    = $gpu->isGPU();
    $ok_shape  = $gpu->shape() === [4];
    $ok_values = (string)$back === (string)$cpu;
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' values=', ($ok_values ? 'OK' : 'BAD'),
         "\n";
}

/* On-device arithmetic — fill(5) + fill(5) must stay on GPU and yield 10. */
$g = NumPower::full([3, 3], 5, 'int32', NUMPOWER_CUDA);
$h = NumPower::full([3, 3], 5, 'int32', NUMPOWER_CUDA);
$sum = NumPower::add($g, $h);
echo 'gpu_arith_sum_eq_90: ',
     ((string)NumPower::sum($sum) === '90' && $sum->isGPU() ? 'OK' : 'BAD'), "\n";

/* String-form fill on GPU for the wide dtypes — the only loss-free
   route for fp128 / int64 / uint64 boundaries. */
$gv = NumPower::full([2], '18446744073709551615', 'uint64', NUMPOWER_CUDA);
$expect = NumPower::full([2], '18446744073709551615', 'uint64', NUMPOWER_CPU);
echo 'gpu_uint64_max_str: ',
     ((string)$gv->cpu() === (string)$expect ? 'OK' : 'BAD'), "\n";

$gv = NumPower::full([2], '-9223372036854775808', 'int64', NUMPOWER_CUDA);
$expect = NumPower::full([2], '-9223372036854775808', 'int64', NUMPOWER_CPU);
echo 'gpu_int64_min_str: ',
     ((string)$gv->cpu() === (string)$expect ? 'OK' : 'BAD'), "\n";

/* Broadcast sizes around the doubling-loop boundaries. */
foreach ([1, 7, 8, 9, 1024, 1025] as $n) {
    $a = NumPower::full([$n], 3, 'float64', NUMPOWER_CUDA);
    $ok = ((string)NumPower::sum($a) === (string)($n * 3));
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
gpu_arith_sum_eq_90: OK
gpu_uint64_max_str: OK
gpu_int64_min_str: OK
gpu_n=1: OK
gpu_n=7: OK
gpu_n=8: OK
gpu_n=9: OK
gpu_n=1024: OK
gpu_n=1025: OK
