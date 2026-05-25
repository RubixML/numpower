--TEST--
NumPower::poisson($shape, $lam, $dtype, NUMPOWER_CUDA) allocates in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* poisson() with device=1 must build the array on the GPU without
   host staging of the result. cuRAND's `curandGeneratePoisson` writes
   uint32 samples directly into a VRAM scratch (or the destination for
   uint32 dtype); the per-dtype path then casts via `cuda_cast_u32_*`
   into the typed destination. fp128 uses `cuda_cast_u32_to_dd` so the
   integer count widens into (hi, lo) in one kernel. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    if ($dt === 'float4')      { $lam = 1.0; }
    elseif ($dt === 'float8')  { $lam = 2.0; }
    elseif ($dt === 'int8')    { $lam = 30.0; }
    elseif ($dt === 'uint8')   { $lam = 100.0; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $lam = 1000.0; }
    elseif ($dt === 'uint64')  { $lam = '100000'; }
    elseif ($dt === 'float128'){ $lam = '50.0'; }
    else                       { $lam = 5.0; }

    $g = NumPower::poisson([8], $lam, $dt, NUMPOWER_CUDA);
    $cpu_back = $g->cpu();
    $ok_dev       = $g->isGPU();
    $ok_shape     = $g->shape() === [8];
    $ok_cpu_shape = $cpu_back->shape() === [8];
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' cpu_back_shape=', ($ok_cpu_shape ? 'OK' : 'BAD'),
         "\n";
}

/* Distribution check on GPU float32: mean / variance must come out
   close to λ. cuRAND's Poisson is itself well-tested so this is just
   a sanity check that we wired the rate correctly. */
function php_mean_var_gpu($a) {
    $arr = $a->cpu()->toArray();
    $sum = 0.0;
    $n   = 0;
    foreach ($arr as $v) {
        $sum += (float)$v;
        $n++;
    }
    $m = $sum / $n;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $m; $sq += $d * $d; }
    return [$m, $sq / $n];
}

$n = 8192;
$g = NumPower::poisson([$n], 100.0, 'float32', NUMPOWER_CUDA);
[$mean, $var] = php_mean_var_gpu($g);
$mean_ok = abs($mean - 100.0) < 2.0;
$var_ok  = abs($var  - 100.0) < 5.0;
echo 'dist_gpu_f32_lam100: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* GPU large λ — cuRAND's internal PTRS-like algorithm handles this. */
$g = NumPower::poisson([$n], 10000.0, 'int32', NUMPOWER_CUDA);
[$mean, $var] = php_mean_var_gpu($g);
$mean_ok = abs($mean - 10000.0) < 20.0;
/* Variance of the sample variance is ~2σ⁴/N. For σ²=10000 and N=8192
   that's stddev ≈ 156; allow a ~4σ window so the test passes
   essentially always while still catching a path that produces a
   wildly different distribution. */
$var_ok  = abs($var  - 10000.0) < 600.0;
echo 'dist_gpu_lam10000: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* cuRAND's Poisson has an internal precision bound that the empirical
   driver enforces somewhere above lam ≈ 4 × 10^5 (XORWOW backing). We
   surface that as a clear PHP error rather than returning a zeroed
   buffer; the user can re-issue on the CPU device for very large
   rates. */
try {
    NumPower::poisson([16], 1.0e9, 'uint32', NUMPOWER_CUDA);
    echo 'curand_lam_reject: BAD (no throw)', "\n";
} catch (\Error $e) {
    echo 'curand_lam_reject: ',
         (str_contains($e->getMessage(),
                       "cuRAND rejected") ? 'OK' : "BAD($e)"),
         "\n";
}

/* Direct uint32 path — no scratch, cuRAND writes straight into the
   destination. */
$g = NumPower::poisson([$n], 50.0, 'uint32', NUMPOWER_CUDA);
[$mean, $var] = php_mean_var_gpu($g);
$mean_ok = abs($mean - 50.0) < 2.0;
echo 'dist_gpu_u32_lam50: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"), "\n";

/* GPU uint64 — VRAM-direct via cuda_cast_u32_to_u64 (no host
   staging of the result). Counts strictly within ~6σ of lam. We pick
   lam = 100000 (safely inside cuRAND's empirical precision bound). */
$g = NumPower::poisson([1024], '100000', 'uint64', NUMPOWER_CUDA);
$arr = $g->cpu()->toArray();
$range_ok = true;
foreach ($arr as $v) {
    $iv = (int)$v;
    /* 6σ window: ±6 * sqrt(1e5) ≈ ±1897. Widen to ±3000 for safety. */
    if ($iv < 97000 || $iv > 103000) { $range_ok = false; break; }
}
echo 'uint64_gpu_dist_range: ', ($range_ok && $g->isGPU() ? 'OK' : 'BAD'), "\n";

/* GPU float128 — DD widen kernel. Spot-check the basic contract. */
$g = NumPower::poisson([16], '50.0', 'float128', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'fp128_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* GPU int64. */
$g = NumPower::poisson([16], 1000, 'int64', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [16];
echo 'int64_gpu_basic: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Multi-dim on GPU. */
$g = NumPower::poisson([16, 64], 5.0, 'float32', NUMPOWER_CUDA);
echo 'multidim_gpu: shape=', implode('x', $g->shape()),
     ' device=', ($g->isGPU() ? 'gpu' : 'cpu'), "\n";

/* 0-D shape on the GPU — must still be an NDArray, not a primitive. */
$g = NumPower::poisson([], 5.0, 'float32', NUMPOWER_CUDA);
$ok = ($g instanceof NDArray) && $g->isGPU() && $g->shape() === [];
echo '0d_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";

/* [0] shape on the GPU — empty array; no kernel launch. */
$g = NumPower::poisson([0], 5.0, 'float32', NUMPOWER_CUDA);
$ok = $g->isGPU() && $g->shape() === [0] && $g->size() === 0;
echo 'zero_size_gpu: ', ($ok ? 'OK' : 'BAD'), "\n";

/* lam = 0 on GPU — every sample must be 0. */
$g = NumPower::poisson([64], 0.0, 'int32', NUMPOWER_CUDA);
$all_zero = true;
foreach ($g->cpu()->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'lam_zero_gpu: ', ($all_zero ? 'OK' : 'BAD'), "\n";
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
dist_gpu_f32_lam100: mean_ok=OK var_ok=OK
dist_gpu_lam10000: mean_ok=OK var_ok=OK
curand_lam_reject: OK
dist_gpu_u32_lam50: mean_ok=OK
uint64_gpu_dist_range: OK
fp128_gpu_basic: OK
int64_gpu_basic: OK
multidim_gpu: shape=16x64 device=gpu
0d_gpu: OK
zero_size_gpu: OK
lam_zero_gpu: OK
