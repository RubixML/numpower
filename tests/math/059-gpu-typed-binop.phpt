--TEST--
GPU + GPU arithmetic stays on GPU for every supported dtype (no CPU fallback)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* User-visible contract: when both operands are on GPU, the result is on GPU.
   No silent CPU round-trip for non-float32 dtypes. This locks in the typed
   GPU kernels (int8/uint8/.../uint64, float16, float64) and the double-double
   emulation kernel for float128. */

$dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
           'float16', 'float32', 'float64', 'float128'];

foreach ($dtypes as $dt) {
    $a = NumPower::array([4, 4, 4, 4], $dt)->gpu();
    $b = NumPower::array([2, 2, 2, 2], $dt)->gpu();

    $sum = $a + $b;       /* 4+2 = 6 */
    $sub = $a - $b;       /* 4-2 = 2 */
    $mul = $a * $b;       /* 4*2 = 8 */
    $div = $a / $b;       /* 4/2 = 2 (or 2.0 if int → float) */
    $pow = $a ** $b;      /* 4**2 = 16 */

    /* Every operation must keep the result on GPU. */
    $all_gpu = $sum->isGPU() && $sub->isGPU() && $mul->isGPU()
               && $div->isGPU() && $pow->isGPU();
    /* And the values must be correct. */
    $vals_ok = (string)$sum->cpu()[0] === '6'
            && (string)$sub->cpu()[0] === '2'
            && (string)$mul->cpu()[0] === '8'
            && (string)$pow->cpu()[0] === '16';
    echo "$dt: gpu=", ($all_gpu ? 'OK' : 'BAD'),
         " values=", ($vals_ok ? 'OK' : 'BAD'),
         " div=", (string)$div->cpu()[0], "\n";
}
?>
--EXPECT--
int8: gpu=OK values=OK div=2
uint8: gpu=OK values=OK div=2
int16: gpu=OK values=OK div=2
uint16: gpu=OK values=OK div=2
int32: gpu=OK values=OK div=2
uint32: gpu=OK values=OK div=2
int64: gpu=OK values=OK div=2
uint64: gpu=OK values=OK div=2
float16: gpu=OK values=OK div=2
float32: gpu=OK values=OK div=2
float64: gpu=OK values=OK div=2
float128: gpu=OK values=OK div=2
