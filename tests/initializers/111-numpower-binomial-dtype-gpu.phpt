--TEST--
NumPower::randomBinomial($shape, $n, $p, $dtype, NUMPOWER_CUDA) allocates in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* randomBinomial() with device=1 must build the array on the GPU
   without host staging of the result. A custom per-thread cuRAND
   kernel does n Bernoulli trials per output slot and writes the
   count as uint32; the per-dtype path then casts via
   `cuda_cast_u32_*` into the typed destination. fp128 uses
   `cuda_cast_u32_to_dd` so the integer count widens into (hi, lo)
   in one kernel. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    if ($dt === 'float4')      { $n_trials = 1; }
    elseif ($dt === 'float8')  { $n_trials = 2; }
    elseif ($dt === 'int8')    { $n_trials = 100; }
    elseif ($dt === 'uint8')   { $n_trials = 200; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $n_trials = 1000; }
    else                       { $n_trials = 50; }

    $g = NumPower::randomBinomial([8], $n_trials, 0.5, $dt, NUMPOWER_CUDA);
    $cpu_back = $g->cpu();
    $ok_dev       = $g->isGPU();
    $ok_shape     = $g->shape() === [8];
    $ok_cpu_shape = $cpu_back->shape() === [8];
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' cpu_back_shape=', ($ok_cpu_shape ? 'OK' : 'BAD'),
         "\n";
}

/* Distribution check on GPU. mean = np, variance = np(1-p). */
function php_mean_var_gpu($a) {
    $arr = $a->cpu()->toArray();
    $sum = 0.0; $n = 0;
    foreach ($arr as $v) { $sum += (float)$v; $n++; }
    $m = $sum / $n;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $m; $sq += $d * $d; }
    return [$m, $sq / $n];
}

$N = 8192;
$g = NumPower::randomBinomial([$N], 100, 0.5, 'float32', NUMPOWER_CUDA);
[$mean, $var] = php_mean_var_gpu($g);
$mean_ok = abs($mean - 50.0) < 1.0;
$var_ok  = abs($var  - 25.0) < 2.0;
echo 'dist_gpu_B100_0.5: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* GPU B(1000, 0.2) — moderate n, asymmetric p. */
$g = NumPower::randomBinomial([$N], 1000, 0.2, 'int32', NUMPOWER_CUDA);
[$mean, $var] = php_mean_var_gpu($g);
$mean_ok = abs($mean - 200.0) < 2.0;
$var_ok  = abs($var  - 160.0) < 10.0;
echo 'dist_gpu_B1000_0.2: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* Direct uint32 path — no scratch, kernel writes straight into the
   destination. */
$g = NumPower::randomBinomial([$N], 50, 0.4, 'uint32', NUMPOWER_CUDA);
[$mean, $var] = php_mean_var_gpu($g);
echo 'dist_gpu_u32_B50_0.4: mean_ok=', (abs($mean - 20.0) < 1.0 ? 'OK' : "BAD($mean)"), "\n";

/* GPU uint64 — VRAM-direct via cuda_cast_u32_to_u64. Counts strictly
   in `[0, n]`. */
$g = NumPower::randomBinomial([1024], 100, 0.5, 'uint64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$range_ok = true;
foreach ($arr as $v) {
    $iv = (int)$v;
    if ($iv < 0 || $iv > 100) { $range_ok = false; break; }
}
echo 'uint64_gpu_count_range: ', ($range_ok && $g->isGPU() ? 'OK' : 'BAD'), "\n";

/* GPU float128 — DD widen kernel. Spot-check the basic contract. */
$g = NumPower::randomBinomial([16], 50, 0.3, 'float128', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'fp128_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU int64. */
$g = NumPower::randomBinomial([16], 500, 0.5, 'int64', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'int64_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Multi-dim on GPU. */
$g = NumPower::randomBinomial([16, 64], 20, 0.5, 'float32', NUMPOWER_CUDA);
echo 'multidim_gpu: shape=', implode('x', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* 0-D shape on the GPU — must still be an NDArray, not a primitive. */
$g = NumPower::randomBinomial([], 10, 0.5, 'float32', NUMPOWER_CUDA);
$ok = ($g instanceof NDArray) && $g->isGPU() && $g->shape() === [];
echo '0d_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";

/* [0] shape on the GPU — empty array; no kernel launch. */
$g = NumPower::randomBinomial([0], 10, 0.5, 'float32', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [0] && $g->size() === 0;
echo 'zero_size_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";

/* n = 0 on GPU — short-circuited via cudaMemset; every sample must
   be 0. */
$g = NumPower::randomBinomial([64], 0, 0.5, 'int32', NUMPOWER_CUDA);
$all_zero = true;
foreach ($g->cpu()->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'n_zero_gpu_shortcircuit: ', ($all_zero ? 'OK' : 'BAD'), "\n";

/* p = 0 on GPU — also short-circuited (no kernel launch needed). */
$g = NumPower::randomBinomial([64], 50, 0.0, 'int32', NUMPOWER_CUDA);
$all_zero = true;
foreach ($g->cpu()->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'p_zero_gpu_shortcircuit: ', ($all_zero ? 'OK' : 'BAD'), "\n";

/* p = 1.0 on GPU — every sample must be exactly n (the legacy CPU
   bug from `rand() / RAND_MAX = 1.0` doesn't apply to GPU
   `1 - curand_uniform` but the contract still requires correctness). */
$g = NumPower::randomBinomial([256], 50, 1.0, 'int32', NUMPOWER_CUDA);
$all_n = true;
foreach ($g->cpu()->toArray() as $v) {
    if ((int)$v !== 50) { $all_n = false; break; }
}
echo 'p_one_gpu_all_n: ', ($all_n ? 'OK' : 'BAD'), "\n";
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
dist_gpu_B100_0.5: mean_ok=OK var_ok=OK
dist_gpu_B1000_0.2: mean_ok=OK var_ok=OK
dist_gpu_u32_B50_0.4: mean_ok=OK
uint64_gpu_count_range: OK
fp128_gpu_basic: OK
int64_gpu_basic: OK
multidim_gpu: shape=16x64 device=gpu
0d_gpu: OK
zero_size_gpu: OK
n_zero_gpu_shortcircuit: OK
p_zero_gpu_shortcircuit: OK
p_one_gpu_all_n: OK
