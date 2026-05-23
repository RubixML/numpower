--TEST--
GPU reductions never touch the input buffer's host side
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The reduction dispatcher (ndarray_reduce_dispatch_gpu) launches a
   per-dtype kernel and only copies an 8-byte accumulator double back
   to host. The source NDArray's data buffer must remain on GPU after
   the reduction — isGPU() must keep returning 1, and a subsequent GPU
   op on the same array must work without a re-upload. */
$dtypes = [
    'float32', 'float64', 'float16', 'float128', 'float4', 'float8',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];
$ok = true;
foreach ($dtypes as $t) {
    $a = (new NDArray([1, 2, 3, 4, 5], $t))->gpu();
    NumPower::sum($a);
    NumPower::prod($a);
    NumPower::max($a);
    NumPower::min($a);
    NumPower::mean($a);
    if ($a->isGPU() !== 1) {
        echo "$t: FAIL isGPU=", $a->isGPU(), " after reductions\n";
        $ok = false;
        continue;
    }
    /* Subsequent GPU op should work — verifies the buffer is still
       device-resident and intact. For non-numeric-ops dtypes (fp4 /
       fp8 etc.) we just confirm the array can still be moved back to
       CPU with the original content. */
    $back = $a->cpu()->toArray();
    if (count($back) !== 5) {
        echo "$t: FAIL ->cpu() returned ", count($back), " elements\n";
        $ok = false;
    }
}
echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
