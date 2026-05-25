--TEST--
NumPower::truncatedNormal($shape, $loc, $scale, $dtype, NUMPOWER_CUDA) allocates in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* truncatedNormal() with device=1 must build the array on the GPU and
   the on-device bytes must decode through the typed-D2H path used by
   every GPU read. Each DOUBLE-kind dtype runs through
   `cuda_truncated_normal_f32 → cuda_cast_f32_to_<dst>` entirely in VRAM
   (no full host staging of the result). fp128 GPU uses a standardised
   truncated f64 stream + `cuda_normal_dd_affine` for the affine. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    if ($dt === 'float4') { $loc = 1.0; $scale = 0.5; }
    elseif ($dt === 'float8') { $loc = 2.0; $scale = 1.0; }
    elseif ($dt === 'int8') { $loc = 0; $scale = 20; }
    elseif ($dt === 'uint8') { $loc = 128; $scale = 30; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $loc = 1000; $scale = 100; }
    elseif ($dt === 'uint64') { $loc = '1000000'; $scale = '100'; }
    elseif ($dt === 'float128') { $loc = '0.0'; $scale = '1.0'; }
    else { $loc = 0.0; $scale = 1.0; }

    $g = NumPower::truncatedNormal([8], $loc, $scale, $dt, NUMPOWER_CUDA);
    $cpu_back = $g->cpu();
    echo $dt, ': gpu=', ($g->isGPU() ? 1 : 0),
         ' shape=', ($g->shape() === [8] ? 'OK' : 'BAD'),
         ' cpu_back_shape=', ($cpu_back->shape() === [8] ? 'OK' : 'BAD'),
         "\n";
}

/* Truncation window check on GPU float32 with shifted distribution. */
function check_window_gpu($g, $loc, $scale, $tag, $eps = 1e-4) {
    $arr = $g->cpu()->toArray();
    $lo = $loc - 2.0 * $scale - $eps;
    $hi = $loc + 2.0 * $scale + $eps;
    $bad = 0;
    foreach ($arr as $v) {
        $f = (float)$v;
        if ($f < $lo || $f > $hi) $bad++;
    }
    echo $tag, ': out_of_window=', $bad, "\n";
}

$g = NumPower::truncatedNormal([2048], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
check_window_gpu($g, 0.0, 1.0, 'window_gpu_f32_std');

$g = NumPower::truncatedNormal([2048], 5.0, 2.0, 'float64', NUMPOWER_CUDA);
check_window_gpu($g, 5.0, 2.0, 'window_gpu_f64_shifted');

/* Distribution check on GPU float32 — same E[X], Var[X] as CPU. */
function php_std_gpu($g, $mean) {
    $arr = $g->cpu()->toArray();
    $sum = 0.0; $n = 0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sum += $d * $d; $n++; }
    return sqrt($sum / $n);
}

$n = 8192;
$g = NumPower::truncatedNormal([$n], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
$mean = (float) NumPower::mean($g);
$std  = php_std_gpu($g, $mean);
$mean_ok = abs($mean) < 0.1;
$std_ok  = abs($std - 0.8796) < 0.1;
echo 'stat_gpu_f32: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"), "\n";

/* Odd-size GPU. The truncated-normal kernel is per-thread rejection,
   so odd sizes are handled directly (no even-size padding like the
   curand bulk APIs need). */
$g = NumPower::truncatedNormal([1023], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
check_window_gpu($g, 0.0, 1.0, 'window_gpu_odd_1023');
echo 'odd_size_gpu: shape=', implode(',', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Multi-dim. */
$g = NumPower::truncatedNormal([16, 64], 0.0, 1.0, 'float32', NUMPOWER_CUDA);
echo 'multidim_gpu: shape=', implode('x', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* GPU fp128 — DD affine path on device. */
$g = NumPower::truncatedNormal([16], '0.0', '1.0', 'float128', NUMPOWER_CUDA);
echo 'fp128_gpu: gpu=', ($g->isGPU() ? 1 : 0),
     ' shape=', ($g->shape() === [16] ? 'OK' : 'BAD'),
     "\n";

/* GPU int64 — fp64 scratch + cuda_cast_f64_to_i64. */
$g = NumPower::truncatedNormal([16], 0, 1000, 'int64', NUMPOWER_CUDA);
check_window_gpu($g, 0.0, 1000.0, 'window_gpu_i64', 1.0);

/* GPU uint64 — VRAM-direct via cuda_truncated_normal_f64 +
   cuda_normal_u64_affine (no host staging of the result). Still bounded
   to [loc-2σ, loc+2σ] = [999800, 1000200]. Larger n raises the chance
   any rejection-sampling gap surfaces as out-of-window. */
$g = NumPower::truncatedNormal([1024], '1000000', '100', 'uint64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$bad = 0;
foreach ($arr as $v) {
    $u = (int)$v;
    if ($u < 999800 || $u > 1000200) $bad++;
}
echo 'window_gpu_u64: out_of_window=', $bad, "\n";
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
window_gpu_f32_std: out_of_window=0
window_gpu_f64_shifted: out_of_window=0
stat_gpu_f32: mean_ok=OK std_ok=OK
window_gpu_odd_1023: out_of_window=0
odd_size_gpu: shape=1023 device=gpu
multidim_gpu: shape=16x64 device=gpu
fp128_gpu: gpu=1 shape=OK
window_gpu_i64: out_of_window=0
window_gpu_u64: out_of_window=0
