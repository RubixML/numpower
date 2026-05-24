--TEST--
NumPower::diag() on GPU does not leak VRAM across many allocations / dtypes
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress: many diag() allocations across every dtype, both directions
   (1-D → 2-D and 2-D → 1-D), and a few input devices to exercise the
   normalisation path (cast + move) inside `ndarray_diag_prepare_input`.
   Any imbalance surfaces as `VRAM MEMORY LEAK: leaked N array(s)` at
   RSHUTDOWN. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $dt) {
    for ($i = 0; $i < 3; $i++) {
        /* 1-D → 2-D on GPU (input on CPU, will be moved). */
        $v   = NumPower::arange(8, 0, 1, $dt);
        $m   = NumPower::diag($v, $dt, NUMPOWER_CUDA);
        $cpu = $m->cpu();
        unset($v, $m, $cpu);

        /* 1-D → 2-D on GPU (input already on GPU — no normalisation). */
        $v   = NumPower::arange(8, 0, 1, $dt)->gpu();
        $m   = NumPower::diag($v, $dt, NUMPOWER_CUDA);
        unset($v, $m);

        /* 2-D → 1-D on GPU (input from identity, already on GPU). */
        $eye = NumPower::identity(8, $dt, NUMPOWER_CUDA);
        $d   = NumPower::diag($eye, $dt, NUMPOWER_CUDA);
        unset($eye, $d);

        /* 2-D → 1-D, cross-device move (GPU input, CPU output). */
        $eye = NumPower::identity(8, $dt, NUMPOWER_CUDA);
        $d   = NumPower::diag($eye, $dt, NUMPOWER_CPU);
        unset($eye, $d);

        /* dtype cast inside the prepare step: input float64 → result $dt. */
        if ($dt !== 'float64') {
            $v = NumPower::arange(8, 0, 1, 'float64');
            $m = NumPower::diag($v, $dt, NUMPOWER_CUDA);
            unset($v, $m);
        }
    }
}

echo "done\n";
?>
--EXPECT--
done
