--TEST--
NDArray::count() does not allocate or leak across many invocations
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* count() reads only metadata, so a tight loop over many arrays of every
   dtype/rank must not leave anything behind. The VRAM leak hook at
   RSHUTDOWN prints "VRAM MEMORY LEAK" if it sees an unfreed device buffer. */

$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($dtypes as $t) {
    /* 1-D, 2-D, and 0-D in a loop. */
    for ($i = 0; $i < 200; $i++) {
        $vec = new NDArray([1, 2, 3, 4, 5], $t);
        $mat = new NDArray([[1, 2], [3, 4], [5, 6]], $t);
        $sc  = new NDArray(1, $t);

        /* Exercise both call paths. */
        $a = count($vec);  $b = $vec->count();
        $c = count($mat);  $d = $mat->count();
        $e = count($sc);   $f = $sc->count();

        unset($vec, $mat, $sc);
    }
}

echo "ok\n";
?>
--EXPECT--
ok
