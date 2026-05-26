--TEST--
NumPower::normal($shape, $loc, $scale, $dtype, NUMPOWER_CUDA) allocates in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* normal() with device=1 must build the array on the GPU and the
   on-device bytes must decode through the typed-D2H path used by every
   GPU read. Each dtype runs through `cuRAND → cuda_cast_<src>_to_<dst>`
   (or, for fp128, `cuRAND f64 → cuda_normal_dd_affine`) entirely in
   VRAM (no full host staging of the result). The distribution check
   uses cpu() to read mean/std back. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    /* Same dtype-aware (loc, scale) as the CPU test so the value range
       stays inside the dtype's representable window. */
    if ($dt === 'float4') { $loc = 1.0; $scale = 0.5; }
    elseif ($dt === 'float8') { $loc = 2.0; $scale = 1.0; }
    elseif ($dt === 'int8') { $loc = 0; $scale = 20; }
    elseif ($dt === 'uint8') { $loc = 128; $scale = 30; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $loc = 1000; $scale = 100; }
    elseif ($dt === 'uint64') { $loc = '1000000'; $scale = '100'; }
    elseif ($dt === 'float128') { $loc = '0.0'; $scale = '1.0'; }
    else { $loc = 0.0; $scale = 1.0; }

    $g = NumPower::normal([8], $loc, $scale, $dt, NUMPOWER_CUDA);
    $cpu_back = $g->cpu();
    $ok_dev   = $g->isGPU();
    $ok_shape = $g->shape() === [8];
    $ok_cpu_shape = $cpu_back->shape() === [8];
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' cpu_back_shape=', ($ok_cpu_shape ? 'OK' : 'BAD'),
         "\n";
}

/* Distribution check on GPU float32: mean/std must come out close to
   the requested N(0, 1). cuRAND is itself well-tested so this is a
   sanity check that we wired up loc/scale correctly. NumPower::std
   has a pre-existing dtype bug on float64; we compute std in PHP. */
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
$g = NumPower::normal([$n], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$std  = php_std_gpu($g, $mean);
$mean_ok = (abs($mean) < 0.1);
$std_ok  = (abs($std - 1.0) < 0.1);
echo 'dist_f32_gpu: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* GPU float64 with non-zero loc/scale exercises cuRAND's double-precision
   generator (curandGenerateNormalDouble). */
$g = NumPower::normal([$n], 10.0, 3.0, 'float64', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$std  = php_std_gpu($g, $mean);
$mean_ok = (abs($mean - 10.0) < 0.3);
$std_ok  = (abs($std - 3.0) < 0.3);
echo 'dist_f64_gpu: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* Odd-size buffer: cuRAND requires an even count; the wrapper must
   pad and copy back the requested n. */
$g = NumPower::normal([1023], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$ok_odd = abs($mean) < 0.2 && $g->shape() === [1023];
echo 'odd_size_gpu: ', ($ok_odd ? 'OK' : "BAD m=$mean shape=" . implode(',', $g->shape())), "\n";

/* Multi-dim on GPU. */
$g = NumPower::normal([16, 64], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
echo 'multidim_gpu: shape=', implode('x', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* GPU fp128 — DD affine path on device. */
$g = NumPower::normal([16], '0.0', '1.0', 'float128', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'fp128_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU int64. */
$g = NumPower::normal([16], 0, 1000, 'int64', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'int64_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU uint64 — wide loc string. */
$g = NumPower::normal([16], '18446744073709551000', '10', 'uint64', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'uint64_gpu_wide: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU uint64 — VRAM-direct via cuda_normal_u64_affine (no host
   staging of the result). With loc=1e9, scale=1000 every sample
   should land within ~±6σ = [999994000, 1000006000] in practice. */
$g = NumPower::normal([1024], '1000000000', '1000', 'uint64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$min = PHP_INT_MAX; $max = 0;
foreach ($arr as $v) {
    $iv = (int)$v;
    if ($iv < $min) $min = $iv;
    if ($iv > $max) $max = $iv;
}
$range_ok = ($min > 999990000) && ($max < 1000010000);
echo 'uint64_gpu_dist_range: ', ($range_ok ? 'OK' : "BAD($min..$max)"), "\n";
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
uint64_gpu_wide: OK
uint64_gpu_dist_range: OK
