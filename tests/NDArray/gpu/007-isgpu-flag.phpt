--TEST--
NDArray::isGPU() correctly reflects device placement after gpu()/cpu()
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The original (CPU) array must remain on CPU after gpu() is called — the
   returned object is the GPU copy. Likewise cpu() on a GPU array returns a
   CPU copy and leaves the original in VRAM. */
$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];

foreach ($dtypes as $dtype) {
    $cpu = new NDArray([0, 1, 2, 3], $dtype);
    $gpu = $cpu->gpu();
    $back = $gpu->cpu();

    $orig_still_cpu = $cpu->isGPU() === 0;
    $gpu_is_gpu     = $gpu->isGPU() === 1;
    $gpu_still_gpu  = $gpu->isGPU() === 1;     /* gpu() did not mutate gpu */
    $back_is_cpu    = $back->isGPU() === 0;

    echo $dtype, ': ',
         ($orig_still_cpu && $gpu_is_gpu && $gpu_still_gpu && $back_is_cpu
             ? 'OK' : 'FAIL'),
         "\n";
}
?>
--EXPECT--
float4: OK
float8: OK
float16: OK
float32: OK
float64: OK
float128: OK
int8: OK
uint8: OK
int16: OK
uint16: OK
int32: OK
uint32: OK
int64: OK
uint64: OK
