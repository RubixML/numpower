--TEST--
NumPower::identity($size, $dtype, NUMPOWER_CUDA) allocates directly in VRAM for every dtype
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* identity() with device=1 builds the matrix in VRAM via a zero-init
   (cudaMemset) + a single cudaMemcpy2D writing the diagonal. The matrix
   itself never traverses host memory; only the small per-element seed
   crosses the bus. This test verifies:
    - the on-device bytes decode back to a valid identity (CPU↔GPU parity).
    - on-device arithmetic (identity + identity) stays on GPU.
    - the doubling-broadcast-style sizes (1, 7, 8, 9, 1024, 1025) all
      produce correct traces. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    $gpu = NumPower::identity(5, $dt, NUMPOWER_CUDA);
    $cpu = NumPower::identity(5, $dt);
    $back = $gpu->cpu();
    $ok_dev    = $gpu->isGPU();
    $ok_shape  = $gpu->shape() === [5, 5];
    $ok_values = (string)$back === (string)$cpu;
    $ok_trace  = (string)NumPower::sum($gpu) === '5';
    echo $dt, ': gpu=', ($ok_dev ? 1 : 0),
         ' shape=', ($ok_shape ? 'OK' : 'BAD'),
         ' values=', ($ok_values ? 'OK' : 'BAD'),
         ' trace=', ($ok_trace ? 'OK' : 'BAD'),
         "\n";
}

/* On-device arithmetic — identity(N) + identity(N) keeps shape and stays
   on GPU; trace doubles. */
$a = NumPower::identity(4, 'float32', NUMPOWER_CUDA);
$b = NumPower::identity(4, 'float32', NUMPOWER_CUDA);
$s = NumPower::add($a, $b);
echo 'gpu_arith_trace_eq_2N: ',
     ((string)NumPower::sum($s) === '8' && $s->isGPU() ? 'OK' : 'BAD'), "\n";

/* Sweep sizes around cudaMemcpy2D's row-count parameter to catch any
   off-by-one in the (N+1)*elsize pitch math. */
foreach ([1, 7, 8, 9, 1024, 1025] as $n) {
    $a = NumPower::identity($n, 'float64', NUMPOWER_CUDA);
    $ok = ((string)NumPower::sum($a) === (string)$n);
    echo 'gpu_n=', $n, ': ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Different dtype on a larger size — uint64 trace == size confirms each
   diagonal byte was placed correctly. */
$a = NumPower::identity(100, 'uint64', NUMPOWER_CUDA);
echo 'gpu_100_uint64_trace: ',
     ((string)NumPower::sum($a) === '100' ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float4: gpu=1 shape=OK values=OK trace=OK
float8: gpu=1 shape=OK values=OK trace=OK
float16: gpu=1 shape=OK values=OK trace=OK
float32: gpu=1 shape=OK values=OK trace=OK
float64: gpu=1 shape=OK values=OK trace=OK
float128: gpu=1 shape=OK values=OK trace=OK
int8: gpu=1 shape=OK values=OK trace=OK
uint8: gpu=1 shape=OK values=OK trace=OK
int16: gpu=1 shape=OK values=OK trace=OK
uint16: gpu=1 shape=OK values=OK trace=OK
int32: gpu=1 shape=OK values=OK trace=OK
uint32: gpu=1 shape=OK values=OK trace=OK
int64: gpu=1 shape=OK values=OK trace=OK
uint64: gpu=1 shape=OK values=OK trace=OK
gpu_arith_trace_eq_2N: OK
gpu_n=1: OK
gpu_n=7: OK
gpu_n=8: OK
gpu_n=9: OK
gpu_n=1024: OK
gpu_n=1025: OK
gpu_100_uint64_trace: OK
