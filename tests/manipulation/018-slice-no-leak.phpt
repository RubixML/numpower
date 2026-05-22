--TEST--
Repeated slice() across dtypes does not leak CPU memory or GPU VRAM (both static and instance forms)
--FILE--
<?php
/* NDARRAY_VCHECK=1 instructs the extension to print
   "VRAM MEMORY LEAK: leaked N array(s)" at RSHUTDOWN if any GPU buffer
   survives. We also rely on PHP's own emalloc accounting: leaked CPU
   allocations would surface in the request shutdown reports under
   USE_ZEND_ALLOC=0, but the simpler/more useful canary is the VRAM line.

   This test exercises BOTH the static (non-mutating) and instance (mutating)
   slice paths to make sure neither leaks memory. The instance form swaps
   buffer slots and frees the old NDArray — easy to leak if the swap order
   is wrong. The static form must not retain temporaries it created from
   PHP arrays/scalars. */

$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Error $e) {}

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($types as $t) {
    for ($it = 0; $it < 30; $it++) {
        /* 2D source — 4×4 keeps allocations small but covers strided patterns. */
        $cpu = new NDArray([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ], $t);

        /* Each pattern triggers a different code path inside the C copy:
            row     — fully contiguous run (single bulk memcpy)
            col     — last-axis stride doesn't match elsize (per-row copy)
            sub     — multi-axis range, partially contiguous
            scalar  — 0-D result */
        $row    = NumPower::slice($cpu, 0);
        $col    = NumPower::slice($cpu, [], -1);
        $sub    = NumPower::slice($cpu, [0, 3], [1, 4]);
        $scalar = NumPower::slice($cpu, 2, 2);

        /* Instance form: mutates `$cpu_mut` step by step. The repeated
           buffer-slot swap must release the old NDArray without leaving
           a dangling allocation. */
        $cpu_mut = new NDArray([[1, 2, 3], [4, 5, 6]], $t);
        $cpu_mut->slice(0);
        $cpu_mut->slice([0, 2]);
        unset($cpu_mut);

        if ($has_gpu) {
            $g = $cpu->gpu();
            $row_g    = NumPower::slice($g, 0);
            $col_g    = NumPower::slice($g, [], -1);
            $sub_g    = NumPower::slice($g, [0, 3], [1, 4]);
            $scalar_g = NumPower::slice($g, 2, 2);

            /* GPU instance-mutation path: kernel writes new VRAM, swap
               replaces the slot, old VRAM must be vfree'd. */
            $gpu_mut = (new NDArray([[1, 2, 3], [4, 5, 6]], $t))->gpu();
            $gpu_mut->slice(0);
            $gpu_mut->slice([0, 2]);
            unset($gpu_mut);

            unset($row_g, $col_g, $sub_g, $scalar_g, $g);
        }

        unset($row, $col, $sub, $scalar, $cpu);
    }
}

echo "DONE\n";
?>
--EXPECT--
DONE
