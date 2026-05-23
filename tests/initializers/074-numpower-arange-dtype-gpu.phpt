--TEST--
NumPower::arange($stop, $start, $step, $dtype, NUMPOWER_CUDA) builds in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* arange() with device=1 builds the destination matrix in VRAM and ships
   only an n*elsize host scratch to the device via NDArray_TypedH2D
   (which converts host fp128 to on-device DD for the fp128 case). The
   final bytes on device must decode back to the same values the CPU
   path produces. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    $n = ($dt === 'float4') ? 4 : 8;
    $gpu = NumPower::arange($n, 0, 1, $dt, NUMPOWER_CUDA);
    $cpu = NumPower::arange($n, 0, 1, $dt);
    $back = $gpu->cpu();
    $ok_dev    = $gpu->isGPU();
    $ok_shape  = $gpu->shape() === [$n];
    $ok_values = (string)$back === (string)$cpu;
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' values=', ($ok_values ? 'OK' : 'BAD'),
         "\n";
}

/* On-device arithmetic — adding two GPU-side aranges must stay on GPU. */
$a = NumPower::arange(5, 0, 1, 'float32', NUMPOWER_CUDA);
$b = NumPower::arange(5, 0, 1, 'float32', NUMPOWER_CUDA);
$s = NumPower::add($a, $b);
echo 'gpu_arith: ',
     ($s->isGPU() && (string)NumPower::sum($s) === '20' ? 'OK' : 'BAD'), "\n";

/* String-form wide-dtype boundary on GPU. */
$g = NumPower::arange('18446744073709551615', '18446744073709551610', '1',
                     'uint64', NUMPOWER_CUDA);
$c = NumPower::arange('18446744073709551615', '18446744073709551610', '1',
                     'uint64', NUMPOWER_CPU);
echo 'gpu_uint64_max: ',
     ((string)$g->cpu() === (string)$c ? 'OK' : 'BAD'), "\n";

/* Non-trivial step on GPU. */
$g = NumPower::arange(1.0, 0.0, 0.25, 'float64', NUMPOWER_CUDA);
echo 'gpu_step_0.25: ', (string)$g, "\n";

/* Negative step on GPU. */
$g = NumPower::arange(0, 5, -1, 'int32', NUMPOWER_CUDA);
echo 'gpu_neg_step: ', (string)$g, "\n";

/* Lengths around obvious boundaries. */
foreach ([1, 7, 8, 1024] as $n) {
    $a = NumPower::arange($n, 0, 1, 'float64', NUMPOWER_CUDA);
    $ok = ($a->size() === $n &&
           (string)NumPower::sum($a) === (string)($n * ($n - 1) / 2));
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
gpu_arith: OK
gpu_uint64_max: OK
gpu_step_0.25: [0, 0.25, 0.5, 0.75]
gpu_neg_step: [5, 4, 3, 2, 1]
gpu_n=1: OK
gpu_n=7: OK
gpu_n=8: OK
gpu_n=1024: OK
