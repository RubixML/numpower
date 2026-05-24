--TEST--
NumPower::diag() — both directions build in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* GPU contract:
    - The result is allocated directly in VRAM via NDArray_Zeros / Empty.
    - The diagonal traffic is a single cudaMemcpy2D device-to-device call,
      not a host round-trip.
    - For every dtype the on-device bytes must decode back to the same
      values the CPU path produces. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    /* 1-D → 2-D: GPU result, value-equal to CPU result. */
    $src = NumPower::arange(4, 0, 1, $dt);
    $gpu = NumPower::diag($src, $dt, NUMPOWER_CUDA);
    $cpu = NumPower::diag($src, $dt, NUMPOWER_CPU);
    $values_ok = (string)$gpu->cpu() === (string)$cpu;
    $shape_ok  = $gpu->shape() === [4, 4];
    $dev_ok    = $gpu->isGPU();
    echo "$dt 1d→2d: gpu=", ($dev_ok ? 1 : 0),
         " shape=", ($shape_ok  ? 'OK' : 'BAD'),
         " values=", ($values_ok ? 'OK' : 'BAD'), "\n";

    /* 2-D → 1-D: extract diagonal on GPU. */
    $eye = NumPower::identity(5, $dt);
    $g   = NumPower::diag($eye, $dt, NUMPOWER_CUDA);
    $c   = NumPower::diag($eye, $dt, NUMPOWER_CPU);
    $values_ok = (string)$g->cpu() === (string)$c;
    $shape_ok  = $g->shape() === [5];
    $dev_ok    = $g->isGPU();
    echo "$dt 2d→1d: gpu=", ($dev_ok ? 1 : 0),
         " shape=", ($shape_ok  ? 'OK' : 'BAD'),
         " values=", ($values_ok ? 'OK' : 'BAD'), "\n";
}

/* Non-square 2-D → 1-D on GPU. */
$m = NumPower::array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 'int32');
$d = NumPower::diag($m, 'int32', NUMPOWER_CUDA);
echo "gpu 3x4 diag: ", (string)$d->cpu(), " isGPU=", ($d->isGPU()?1:0), "\n";

/* Cross-device input/output: CPU input → GPU output (1-D → 2-D). */
$src = NumPower::array([1.0, 2.0, 3.0], 'float64');     /* CPU */
$g   = NumPower::diag($src, 'float64', NUMPOWER_CUDA);  /* GPU result */
echo "cpu→gpu 1d→2d trace=",
     ((string)NumPower::sum($g) === '6' && $g->isGPU() ? 'OK' : 'BAD'), "\n";

/* Cross-device input/output: GPU input → CPU output (2-D → 1-D). */
$src = NumPower::array([[1.0, 2.0], [3.0, 4.0]])->gpu();
$c   = NumPower::diag($src, 'float64', NUMPOWER_CPU);
echo "gpu→cpu 2d→1d: ", (string)$c,
     " isGPU=", ($c->isGPU() ? 1 : 0), "\n";

/* On-device arithmetic: diag(arange) + diag(arange) stays on GPU. */
$a = NumPower::diag(NumPower::arange(4, 0, 1, 'float32'), 'float32', NUMPOWER_CUDA);
$b = NumPower::diag(NumPower::arange(4, 0, 1, 'float32'), 'float32', NUMPOWER_CUDA);
$s = NumPower::add($a, $b);
echo "gpu_arith stays on GPU: ",
     ($s->isGPU() && (string)NumPower::sum($s) === '12' ? 'OK' : 'BAD'), "\n";

/* Sweep sizes for the doubling / pitch math (mirrors identity). */
foreach ([1, 7, 8, 1024] as $n) {
    $src = NumPower::arange($n, 0, 1, 'float64');
    $g   = NumPower::diag($src, 'float64', NUMPOWER_CUDA);
    $ok  = ($g->isGPU() &&
            $g->shape() === [$n, $n] &&
            (string)NumPower::sum($g) === (string)($n * ($n - 1) / 2));
    echo "gpu_n=$n: ", ($ok ? 'OK' : 'BAD'), "\n";
}
?>
--EXPECT--
float4 1d→2d: gpu=1 shape=OK values=OK
float4 2d→1d: gpu=1 shape=OK values=OK
float8 1d→2d: gpu=1 shape=OK values=OK
float8 2d→1d: gpu=1 shape=OK values=OK
float16 1d→2d: gpu=1 shape=OK values=OK
float16 2d→1d: gpu=1 shape=OK values=OK
float32 1d→2d: gpu=1 shape=OK values=OK
float32 2d→1d: gpu=1 shape=OK values=OK
float64 1d→2d: gpu=1 shape=OK values=OK
float64 2d→1d: gpu=1 shape=OK values=OK
float128 1d→2d: gpu=1 shape=OK values=OK
float128 2d→1d: gpu=1 shape=OK values=OK
int8 1d→2d: gpu=1 shape=OK values=OK
int8 2d→1d: gpu=1 shape=OK values=OK
uint8 1d→2d: gpu=1 shape=OK values=OK
uint8 2d→1d: gpu=1 shape=OK values=OK
int16 1d→2d: gpu=1 shape=OK values=OK
int16 2d→1d: gpu=1 shape=OK values=OK
uint16 1d→2d: gpu=1 shape=OK values=OK
uint16 2d→1d: gpu=1 shape=OK values=OK
int32 1d→2d: gpu=1 shape=OK values=OK
int32 2d→1d: gpu=1 shape=OK values=OK
uint32 1d→2d: gpu=1 shape=OK values=OK
uint32 2d→1d: gpu=1 shape=OK values=OK
int64 1d→2d: gpu=1 shape=OK values=OK
int64 2d→1d: gpu=1 shape=OK values=OK
uint64 1d→2d: gpu=1 shape=OK values=OK
uint64 2d→1d: gpu=1 shape=OK values=OK
gpu 3x4 diag: [1, 6, 11] isGPU=1
cpu→gpu 1d→2d trace=OK
gpu→cpu 2d→1d: [1, 4] isGPU=0
gpu_arith stays on GPU: OK
gpu_n=1: OK
gpu_n=7: OK
gpu_n=8: OK
gpu_n=1024: OK
