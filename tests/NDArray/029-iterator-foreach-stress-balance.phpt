--TEST--
NDArray Iterator: large sustained foreach keeps PHP memory bounded across all dtypes & ranks
--FILE--
<?php
/* Harder version of 024: many more iterations across every dtype and every
   rank (1-D scalar yields, 2-D NDArray yields). The pre-fix `current()` added
   one buffer slot per iteration, so 14 dtypes × 3 ranks × N iters would have
   leaked ~3*14*N slots into MAIN_MEM_STACK.buffer. We verify the post-fix
   memory delta stays below a comfortable bound.

   The 2-D leg is the real workhorse: each step builds a per-row sub-view that
   goes through ndarray_init_new_object's add_to_buffer path, then the PHP
   object's destructor frees it — exercising the buffer slot reuse via the
   freeList. The 1-D leg covers the scalar zval path. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

/* Warmup. */
foreach ($dtypes as $t) {
    $sIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $v1 = $sIO ? ['1','2','3'] : [1,2,3];
    $v2 = $sIO ? [['1','2'],['3','4']] : [[1,2],[3,4]];
    $a = new NDArray($v1, $t);
    $b = NumPower::array($v2, $t);
    foreach ($a as $x) {}
    foreach ($b as $r) {}
    unset($a, $b);
}

$baseline = memory_get_usage();

/* Stress: 500 iterations × 14 dtypes × 2 ranks = 14000 foreach passes. */
for ($i = 0; $i < 500; $i++) {
    foreach ($dtypes as $t) {
        $sIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $v1 = $sIO ? ['1','2','3'] : [1,2,3];
        $v2 = $sIO ? [['1','2'],['3','4']] : [[1,2],[3,4]];

        $a = new NDArray($v1, $t);
        foreach ($a as $x) {}

        $b = NumPower::array($v2, $t);
        foreach ($b as $r) {
            /* Drive a second-level current to exercise nested view creation. */
            $r->rewind();
            $v = $r->current();
        }
        unset($a, $b);
    }
}

$delta = memory_get_usage() - $baseline;

/* Under the original bug, the buffer-slot leak alone would have grown by
   ~14000 * 14 dtypes ≈ 200K NDArray*-sized slots = several MB. A 512 KB
   ceiling stays well above PHP allocator noise but far below any plausible
   regression. */
if ($delta > 512 * 1024) {
    printf("FAIL: memory grew by %d bytes (>512KB ceiling)\n", $delta);
} else {
    echo "ok\n";
}
?>
--EXPECT--
ok
