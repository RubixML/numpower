--TEST--
NDArray Iterator: foreach on CPU does not grow the global buffer (slot leak)
--FILE--
<?php
/* The old current() called add_to_buffer() on the per-iteration view BEFORE
   ndarray_init_new_object freed it directly for 1-D sources (via the ndim==0
   scalar-zval path). That left a dangling pointer in MAIN_MEM_STACK.buffer
   for every iteration -- a slow leak proportional to total foreach steps.

   We detect that here by tracking process memory across a sustained foreach
   workload. The buffer grew linearly under the bug, so memory growth was
   ~O(iters); after the fix it stays bounded. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

/* Warm-up: lets PHP's allocator settle. */
foreach ($dtypes as $t) {
    $str_io = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $vals = $str_io ? ['1','2','3','4','5'] : [1,2,3,4,5];
    $a = new NDArray($vals, $t);
    foreach ($a as $v) {}
    unset($a);
}

$baseline = memory_get_usage();

/* Heavy workload: 14 dtypes x 200 iters x 5 elements = 14000 foreach steps. */
for ($iter = 0; $iter < 200; $iter++) {
    foreach ($dtypes as $t) {
        $str_io = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $vals = $str_io ? ['1','2','3','4','5'] : [1,2,3,4,5];
        $a = new NDArray($vals, $t);
        foreach ($a as $v) {}
        unset($a);
    }
}

$after = memory_get_usage();
$delta = $after - $baseline;

/* Tolerance: under the bug, delta would scale with 14000 iterations -- well
   over a megabyte. After the fix the growth is small and dominated by PHP's
   own reallocation noise. 200 KB is comfortably above noise but far below
   what the leak produced. */
if ($delta > 200 * 1024) {
    echo "FAIL: memory grew by $delta bytes (suspect buffer slot leak)\n";
} else {
    echo "ok\n";
}
?>
--EXPECT--
ok
