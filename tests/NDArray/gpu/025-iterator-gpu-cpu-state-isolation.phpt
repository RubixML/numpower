--TEST--
NDArray Iterator: GPU/CPU copies have independent cursors; values match across devices
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The iterator is per-NDArray state, so $a->gpu() produces a new NDArray
   with its own cursor at 0. We additionally check that iterating the GPU
   copy and the original CPU copy yields the same sequence — i.e. the GPU
   data transfer (and the dtype-aware scalar read in NDArray_ScalarToZval)
   doesn't perturb values for any dtype. */

$dtypes = [
    'int8','uint8','int16','uint16','int32','uint32','int64','uint64',
    'float4','float8','float16','float32','float64','float128',
];

foreach ($dtypes as $t) {
    $sIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $vals = $sIO ? ['1','2','3','4'] : [1,2,3,4];
    $a = new NDArray($vals, $t);
    $g = $a->gpu();

    /* Walk the CPU side halfway. */
    $a->rewind();
    $a->next();
    $a->next();
    $cpuKey = $a->key();

    /* The GPU copy's cursor is independent. */
    $g->rewind();
    $gpuKey = $g->key();

    /* Compare values pairwise. */
    $cpuVals = [];
    $gpuVals = [];
    foreach ($a as $v) { $cpuVals[] = (string)$v; }
    foreach ($g as $v) { $gpuVals[] = (string)$v; }

    $match = $cpuVals === $gpuVals ? 'OK' : 'BAD';
    echo "$t: cpuKey=$cpuKey gpuKey=$gpuKey vals_match=$match cpu=[",
         implode(',', $cpuVals), "] gpu=[", implode(',', $gpuVals), "]\n";

    unset($a, $g);
}
?>
--EXPECT--
int8: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
uint8: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
int16: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
uint16: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
int32: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
uint32: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
int64: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
uint64: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
float4: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
float8: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
float16: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
float32: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
float64: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
float128: cpuKey=2 gpuKey=0 vals_match=OK cpu=[1,2,3,4] gpu=[1,2,3,4]
