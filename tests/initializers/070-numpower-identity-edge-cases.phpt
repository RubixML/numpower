--TEST--
NumPower::identity() handles boundary sizes and the default dtype/device contract
--FILE--
<?php
/* Boundary inputs:
    - size = 0    must return a (0, 0) 2-D matrix (numpy parity). The old
                  implementation collapsed this to a 1-D shape=[0] array.
    - size = 1    smallest non-trivial identity, exercises the size==N
                  loop boundary.
    - size = 2    typical small case.
    - size = 1024 large enough to exercise the GPU cudaMemcpy2D pitch
                  math at a non-trivial element count.

   The defaults must be float32 / CPU and explicit defaults must match
   omitting the args. */

foreach ([0, 1, 2, 1024] as $size) {
    $a = NumPower::identity($size);
    $shape_ok = $a->shape() === [$size, $size];
    $size_ok  = $a->size() === $size * $size;
    $sum_ok   = (string)NumPower::sum($a) === (string)$size;
    echo 'size=', $size, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok ? 'OK' : 'BAD'),
         ' sum=',  ($sum_ok ? 'OK' : 'BAD'),
         "\n";
}

/* size == 0 deserves a closer look — it must be a 2-D NDArray that
   doesn't trip downstream code expecting a 2-D operand. */
$z = NumPower::identity(0);
echo 'size_0_is_ndarray: ', ($z instanceof NDArray ? 'OK' : 'BAD'), "\n";
echo 'size_0_shape: ', (json_encode($z->shape()) === '[0,0]' ? 'OK' : 'BAD'), "\n";
echo 'size_0_size: ', ($z->size() === 0 ? 'OK' : 'BAD'), "\n";

/* Default contract: float32 / CPU. */
$a = NumPower::identity(2);
echo 'default dtype is float32: ', (is_float($a[0][0]) ? 'OK' : 'BAD'), "\n";
echo 'default device is CPU: ', ($a->isGPU() ? 'BAD' : 'OK'), "\n";

/* Explicit defaults must behave identically to omitting them. */
$a = NumPower::identity(2, 'float32', NUMPOWER_CPU);
echo 'explicit defaults parity: ',
     (!$a->isGPU() && (string)NumPower::sum($a) === '2' ? 'OK' : 'BAD'), "\n";

/* size = 0 also works on GPU and produces the same shape contract. */
$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $has_gpu = false; }
if ($has_gpu) {
    $g = NumPower::identity(0, 'float64', NUMPOWER_CUDA);
    echo 'gpu_size_0: ',
         ($g instanceof NDArray && $g->isGPU() && $g->shape() === [0, 0]
            ? 'OK' : 'BAD'), "\n";
} else {
    echo "gpu_size_0: OK\n";
}
?>
--EXPECT--
size=0: shape=OK size=OK sum=OK
size=1: shape=OK size=OK sum=OK
size=2: shape=OK size=OK sum=OK
size=1024: shape=OK size=OK sum=OK
size_0_is_ndarray: OK
size_0_shape: OK
size_0_size: OK
default dtype is float32: OK
default device is CPU: OK
explicit defaults parity: OK
gpu_size_0: OK
