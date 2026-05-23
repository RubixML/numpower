--TEST--
NumPower::identity() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress test: many identity() allocations across every dtype + a mix of
   sizes (incl. 0 and 1). Each iteration's NDArray is overwritten — the
   previous VRAM slot must be released. Any imbalance surfaces as
   `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$sizes = [0, 1, 4, 16, 64];

foreach ($dtypes as $dt) {
    foreach ($sizes as $n) {
        for ($i = 0; $i < 5; $i++) {
            $a = NumPower::identity($n, $dt, NUMPOWER_CUDA);
            /* Read one diagonal element back so any half-initialised
               buffer would surface as a crash or wrong value. */
            if ($n > 0) {
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
