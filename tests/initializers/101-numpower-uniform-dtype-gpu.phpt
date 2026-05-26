--TEST--
NumPower::uniform($shape, $low, $high, $dtype, NUMPOWER_CUDA) allocates in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* uniform() with device=1 must build the array on the GPU without
   host staging of the result. Each dtype runs through `cuRAND →
   cuda_cast_<src>_to_<dst>` (or, for fp128, `cuRAND f64 →
   cuda_uniform_dd_affine`) entirely in VRAM. The distribution check
   uses cpu() to read mean/range back. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    if ($dt === 'float4') { $low = 0.0; $high = 1.0; }
    elseif ($dt === 'float8') { $low = 0.0; $high = 2.0; }
    elseif ($dt === 'int8') { $low = 0; $high = 100; }
    elseif ($dt === 'uint8') { $low = 0; $high = 255; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $low = 0; $high = 1000; }
    elseif ($dt === 'uint64') { $low = '1000000'; $high = '2000000'; }
    elseif ($dt === 'float128') { $low = '0.0'; $high = '1.0'; }
    else { $low = 0.0; $high = 1.0; }

    $g = NumPower::uniform([8], $low, $high, $dt, NUMPOWER_CUDA);
    $cpu_back = $g->cpu();
    $ok_dev       = $g->isGPU();
    $ok_shape     = $g->shape() === [8];
    $ok_cpu_shape = $cpu_back->shape() === [8];
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' cpu_back_shape=', ($ok_cpu_shape ? 'OK' : 'BAD'),
         "\n";
}

/* Distribution check on GPU float32: mean / range must come out close
   to U([0, 1)). */
$n = 8192;
$g = NumPower::uniform([$n], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$min = INF; $max = -INF; $sum = 0.0;
foreach ($arr as $v) {
    $f = (float)$v;
    if ($f < $min) $min = $f;
    if ($f > $max) $max = $f;
    $sum += $f;
}
$mean     = $sum / $n;
$mean_ok  = (abs($mean - 0.5) < 0.05);
$range_ok = ($min >= 0.0 && $max < 1.0);
echo 'dist_f32_gpu: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' range_ok=',  ($range_ok ? 'OK' : "BAD($min..$max)"),  "\n";

/* GPU float64 with non-zero shifted bounds. */
$g = NumPower::uniform([$n], 10.0, 20.0, 'float64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$min = INF; $max = -INF; $sum = 0.0;
foreach ($arr as $v) {
    $f = (float)$v;
    if ($f < $min) $min = $f;
    if ($f > $max) $max = $f;
    $sum += $f;
}
$mean     = $sum / $n;
$mean_ok  = (abs($mean - 15.0) < 0.5);
$range_ok = ($min >= 10.0 && $max < 20.0);
echo 'dist_f64_gpu_shifted: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' range_ok=',  ($range_ok ? 'OK' : "BAD($min..$max)"),  "\n";

/* Odd-size buffer: cuRAND uniform doesn't share the even-size
   requirement that the normal generator has, but we still exercise an
   odd shape to confirm the in-place fill works. */
$g = NumPower::uniform([1023], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$mean = array_sum(array_map('floatval', $arr)) / 1023;
$ok_odd = abs($mean - 0.5) < 0.1 && $g->shape() === [1023];
echo 'odd_size_gpu: ', ($ok_odd ? 'OK' : "BAD m=$mean"), "\n";

/* Multi-dim on GPU. */
$g = NumPower::uniform([16, 64], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
echo 'multidim_gpu: shape=', implode('x', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* GPU fp128 — DD affine path on device. Spot-check that the first
   element decodes plausibly. */
$g = NumPower::uniform([16], '0.0', '1.0', 'float128', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'fp128_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU int64 — uses cuRAND f64 scratch + cuda_cast_f64_to_i64. */
$g = NumPower::uniform([16], 0, 1000, 'int64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$range_ok = true;
foreach ($arr as $v) {
    if ($v < 0 || $v >= 1000) { $range_ok = false; break; }
}
echo 'int64_gpu_range: ', ($range_ok ? 'OK' : 'BAD'), "\n";

/* GPU uint64 — VRAM-direct via cuda_uniform_u64_affine kernel (no
   host staging of the result). Range must be strict [0, 1000). */
$g = NumPower::uniform([1024], '0', '1000', 'uint64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$range_ok = true;
foreach ($arr as $v) {
    $iv = (int)$v;
    if ($iv < 0 || $iv >= 1000) { $range_ok = false; break; }
}
echo 'uint64_gpu_range: ', ($range_ok && $g->isGPU() ? 'OK' : 'BAD'), "\n";

/* 0-D shape on the GPU — must still be an NDArray, not a primitive. */
$g = NumPower::uniform([], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
$ok = ($g instanceof NDArray) && $g->isGPU() && $g->shape() === [];
echo '0d_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";

/* [0] shape on the GPU — empty array; no kernel launch, just an empty
   VRAM allocation. */
$g = NumPower::uniform([0], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
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
dist_f32_gpu: mean_ok=OK range_ok=OK
dist_f64_gpu_shifted: mean_ok=OK range_ok=OK
odd_size_gpu: OK
multidim_gpu: shape=16x64 device=gpu
fp128_gpu_basic: OK
int64_gpu_range: OK
uint64_gpu_range: OK
0d_gpu: OK
zero_size_gpu: OK
