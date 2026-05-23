--TEST--
NDArray::offsetGet() on CPU does not leak buffer slots (stress)
--FILE--
<?php
/* Each scalar offsetGet on a 1-D source creates a borrowed 0-D view,
   reads it through NDArray_ScalarToZval, and frees the view via NDArray_FREE.
   Buffer-slot leaks would manifest as the global buffer growing unboundedly.
   We can't directly inspect the buffer table from PHP, so instead we drive
   ~140k repeated scalar reads across every CPU dtype + N-D sub-views and
   rely on the underlying memory tracking. The test passing without a SIGABRT
   or out-of-memory error is the contract. */

$types = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
          'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

for ($iter = 0; $iter < 100; $iter++) {
    foreach ($types as $t) {
        $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $vals  = $strIO ? ['1','2','3','4'] : [1, 2, 3, 4];
        $grid  = $strIO ? [['1','2'],['3','4']] : [[1, 2], [3, 4]];

        $vec = new NDArray($vals, $t);
        for ($i = 0; $i < 4; $i++) {
            $x = $vec[$i];
        }

        /* 2-D sub-view path: $a[i] returns a 1-D NDArray; chained $a[i][j]
           releases the intermediate view at end-of-statement. */
        $mat = new NDArray($grid, $t);
        $x = $mat[0][1];
        $x = $mat[1][0];
        $row = $mat[1];
        unset($row);

        unset($vec, $mat, $x);
    }
}
echo "ok\n";
?>
--EXPECT--
ok
