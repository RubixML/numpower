--TEST--
NDArray Iterator: GPU 0-D source -- valid() is false, current() is null
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* 0-D parity with CPU: the iterator must short-circuit on 0-D GPU NDArrays
   too, never reading the missing shape[0] or trying to issue a cudaMemcpy
   from an offset off the end of VRAM.

   We cannot use `(new NDArray(7, $t))->gpu()` here because NDArray::gpu()
   collapses a 0-D source into a dtype-correct PHP scalar (matching the
   long-standing legacy contract). To get a 0-D NDArray that lives on the
   GPU we wrap a single-element 1-D array in gpu(), then slice it down to
   one element (which keeps it on the device). */

$dtypes = ['float32', 'float64', 'float128', 'int8', 'int64', 'uint64'];

foreach ($dtypes as $t) {
    $str_io = in_array($t, ['float128','int64','uint64'], true);
    $vals = $str_io ? ['7'] : [7];
    /* Move the 1-D array to GPU, then squeeze to 0-D in place. */
    $g = (new NDArray($vals, $t))->gpu();
    $g->slice(0);

    $g->rewind();
    echo "$t: ndim=", count($g->shape()),
         " isGPU=", $g->isGPU() ? '1' : '0',
         " valid=", $g->valid() ? '1' : '0',
         " current=", $g->current() === null ? 'null' : 'NOTNULL',
         " key=", $g->key(), "\n";

    $iters = 0;
    foreach ($g as $v) { $iters++; }
    echo "$t foreach iters=$iters\n";
}
?>
--EXPECT--
float32: ndim=0 isGPU=1 valid=0 current=null key=0
float32 foreach iters=0
float64: ndim=0 isGPU=1 valid=0 current=null key=0
float64 foreach iters=0
float128: ndim=0 isGPU=1 valid=0 current=null key=0
float128 foreach iters=0
int8: ndim=0 isGPU=1 valid=0 current=null key=0
int8 foreach iters=0
int64: ndim=0 isGPU=1 valid=0 current=null key=0
int64 foreach iters=0
uint64: ndim=0 isGPU=1 valid=0 current=null key=0
uint64 foreach iters=0
