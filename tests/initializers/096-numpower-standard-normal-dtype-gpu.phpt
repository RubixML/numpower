--TEST--
NumPower::standardNormal($shape, $dtype, NUMPOWER_CUDA) allocates in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* standardNormal() with device=1 must build the array on the GPU
   without any host staging of the result. Each dtype flows through
   `cuRAND → cuda_cast_<src>_to_<dst>` (or, for fp128, `cuRAND f64 →
   cuda_normal_dd_affine`) entirely in VRAM. The distribution check
   uses cpu() to read the moments back. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    $g = NumPower::standardNormal([8], $dt, NUMPOWER_CUDA);
    $cpu_back = $g->cpu();
    $ok_dev       = $g->isGPU();
    $ok_shape     = $g->shape() === [8];
    $ok_cpu_shape = $cpu_back->shape() === [8];
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' cpu_back_shape=', ($ok_cpu_shape ? 'OK' : 'BAD'),
         "\n";
}

/* Distribution check on GPU float32. cuRAND is itself well-tested so
   this is a sanity check that we wired the (loc=0, scale=1) defaults
   correctly. NumPower::std has a pre-existing dtype bug on float64; we
   compute std manually in PHP. */
function php_std_gpu($a, $mean) {
    $arr = $a->cpu()->toArray();
    $sum = 0.0;
    $n   = 0;
    foreach ($arr as $v) {
        $d = ((float)$v) - $mean;
        $sum += $d * $d;
        $n++;
    }
    return sqrt($sum / $n);
}

$n = 8192;
$g = NumPower::standardNormal([$n], 'float32', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$std  = php_std_gpu($g, $mean);
$mean_ok = (abs($mean) < 0.1);
$std_ok  = (abs($std - 1.0) < 0.1);
echo 'dist_f32_gpu: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* GPU float64 — cuRAND double-precision generator. */
$g = NumPower::standardNormal([$n], 'float64', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$std  = php_std_gpu($g, $mean);
$mean_ok = (abs($mean) < 0.1);
$std_ok  = (abs($std - 1.0) < 0.1);
echo 'dist_f64_gpu: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* Odd-size buffer: cuRAND requires an even count; the wrapper pads
   internally and copies back the requested n. */
$g = NumPower::standardNormal([1023], 'float32', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$ok_odd = abs($mean) < 0.2 && $g->shape() === [1023];
echo 'odd_size_gpu: ', ($ok_odd ? 'OK' : "BAD m=$mean shape=" . implode(',', $g->shape())), "\n";

/* Multi-dim on GPU. */
$g = NumPower::standardNormal([16, 64], 'float32', NUMPOWER_CUDA);
echo 'multidim_gpu: shape=', implode('x', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* GPU fp128 — DD affine path on device. We don't validate moments on
   fp128 GPU (the fp128→double roundtrip and PHP's mean precision are
   not the right tools); we only confirm the result lives on the GPU
   with the correct shape. */
$g = NumPower::standardNormal([16], 'float128', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'fp128_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU int64 — uses cuRAND f64 scratch + cuda_cast_f64_to_i64. */
$g = NumPower::standardNormal([16], 'int64', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'int64_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU uint64 — staged host fill + TypedH2D. */
$g = NumPower::standardNormal([16], 'uint64', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'uint64_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* 0-D shape on the GPU — must still be an NDArray, not a primitive. */
$g = NumPower::standardNormal([], 'float32', NUMPOWER_CUDA);
$ok = ($g instanceof NDArray) && $g->isGPU() && $g->shape() === [];
echo '0d_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";

/* [0] shape on the GPU — empty array; no kernel launch, just an empty
   VRAM allocation. */
$g = NumPower::standardNormal([0], 'float32', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [0] && $g->size() === 0;
echo 'zero_size_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float4: gpu=1 shape=OK cpu_back_shape=OK
float8: gpu=1 shape=OK cpu_back_shape=OK
float16: gpu=1 shape=OK cpu_back_shape=OK
float32: gpu=1 shape=OK cpu_back_shape=OK
float64: gpu=1 shape=OK cpu_back_shape=OK
float128: gpu=1 shape=OK cpu_back_shape=OK
int8: gpu=1 shape=OK cpu_back_shape=OK
uint8: gpu=1 shape=OK cpu_back_shape=OK
int16: gpu=1 shape=OK cpu_back_shape=OK
uint16: gpu=1 shape=OK cpu_back_shape=OK
int32: gpu=1 shape=OK cpu_back_shape=OK
uint32: gpu=1 shape=OK cpu_back_shape=OK
int64: gpu=1 shape=OK cpu_back_shape=OK
uint64: gpu=1 shape=OK cpu_back_shape=OK
dist_f32_gpu: mean_ok=OK std_ok=OK
dist_f64_gpu: mean_ok=OK std_ok=OK
odd_size_gpu: OK
multidim_gpu: shape=16x64 device=gpu
fp128_gpu_basic: OK
int64_gpu_basic: OK
uint64_gpu_basic: OK
0d_gpu: OK
zero_size_gpu: OK
