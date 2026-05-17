--TEST--
NDArray::gpu()/cpu() is a no-op when the array is already on the target device
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Per the documented semantics, calling gpu() on an array that already lives
   in VRAM (or cpu() on an array that already lives in host RAM) must perform
   *no* allocation and *no* memory transfer. Object identity is the strongest
   possible proof that no copy was made — the returned zval IS the source. */
$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];

foreach ($dtypes as $dtype) {
    /* CPU -> cpu() is a no-op. */
    $a = new NDArray([0, 1, 2, 3], $dtype);
    $a2 = $a->cpu();
    $cpu_noop = ($a === $a2) && (spl_object_id($a) === spl_object_id($a2));

    /* GPU -> gpu() is a no-op. The first gpu() is a real CPU->GPU transfer;
       the second and third must return the very same PHP object back, and
       repeated calls must keep returning that same object — proving no new
       NDArray was allocated in the buffer for either same-device call. */
    $g  = $a->gpu();
    $g2 = $g->gpu();
    $g3 = $g2->gpu()->gpu()->gpu();
    $gpu_noop = ($g === $g2) && ($g === $g3)
        && (spl_object_id($g) === spl_object_id($g3));

    /* The cross-device hops still produce *new* arrays with their own ids. */
    $real_to_gpu = ($a !== $g) && (spl_object_id($a) !== spl_object_id($g));
    $h = $g->cpu();
    $real_to_cpu = ($g !== $h) && (spl_object_id($g) !== spl_object_id($h));

    echo $dtype, ': ',
         ($cpu_noop && $gpu_noop && $real_to_gpu && $real_to_cpu ? 'OK' : 'FAIL'),
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
