--TEST--
NumPower::arange() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress test: many arange() allocations across every dtype + a mix of
   parameters. Each iteration's NDArray is overwritten — the previous
   VRAM slot must be released. Any imbalance surfaces as
   `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN.

   The wide-dtype string paths (fp128 / int64 / uint64) cover the
   strtoflt128 / strtoull / strtoll branches inside the coercion
   helpers so a refcount imbalance there is also caught. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$cases = [
    [8, 0, 1],
    [0, 0, 1],   /* empty */
    [16, 0, 2],
    [5, 0, -1],  /* sign mismatch → empty */
];

foreach ($dtypes as $dt) {
    $string_io = in_array($dt, ['float128', 'int64', 'uint64'], true);
    foreach ($cases as [$stop, $start, $step]) {
        for ($i = 0; $i < 3; $i++) {
            /* For uint64, skip negative-step variants which we expect
               to reject via the non-negative guard. */
            if ($dt === 'uint64' && $step < 0) continue;
            $a = $string_io
                ? NumPower::arange((string)$stop, (string)$start, (string)$step,
                                   $dt, NUMPOWER_CUDA)
                : NumPower::arange($stop, $start, $step, $dt, NUMPOWER_CUDA);
            if ($a->size() > 0) {
                $cpu = $a->cpu();
                $cpu = null;
            }
            $a = null;
        }
    }
}

echo "done\n";
?>
--EXPECT--
done
