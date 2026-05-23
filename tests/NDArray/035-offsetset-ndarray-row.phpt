--TEST--
NDArray::offsetSet() — `$a[i] = NDArray | array` copies a whole row, dtype-preserving
--FILE--
<?php
/* The non-scalar offsetSet path wraps the value with ZVAL_TO_NDARRAY() and
   delegates to NDArray_Overwrite, which fast-paths the same-dtype case with
   memcpy. Verify across dtypes that:
     - same-dtype NDArray source overwrites the row exactly
     - PHP-array source overwrites the row (cross-dtype casts through double
       for non-float32 targets are tolerated within dtype precision)
     - shape mismatch throws
     - the destination shape and dtype are unchanged after the write */

$dtypes = ['float32', 'float64', 'int32', 'int64'];

foreach ($dtypes as $t) {
    /* Same-dtype NDArray assignment. */
    $a = new NDArray([[1, 2, 3], [4, 5, 6]], $t);
    $row = new NDArray([7, 8, 9], $t);
    $a[0] = $row;
    $got = $a->toArray();
    $ok_nda = $got[0] == [7, 8, 9] && $got[1] == [4, 5, 6];
    /* Source NDArray is intact after the write (no refcount stealing). */
    $src_intact = $row->toArray() == [7, 8, 9];

    /* PHP-array assignment. */
    $b = new NDArray([[1, 2, 3], [4, 5, 6]], $t);
    $b[1] = [70, 80, 90];
    $got2 = $b->toArray();
    $ok_arr = $got2[0] == [1, 2, 3] && $got2[1] == [70, 80, 90];

    /* Shape mismatch throws. */
    $err = 'NONE';
    try {
        $c = new NDArray([[1, 2, 3], [4, 5, 6]], $t);
        $c[0] = new NDArray([99, 100], $t);
    } catch (\Throwable $e) {
        $err = 'THROWN';
    }

    echo "$t: nda=", $ok_nda ? "OK" : "BAD",
         " src_intact=", $src_intact ? "OK" : "BAD",
         " arr=", $ok_arr ? "OK" : "BAD",
         " mismatch=$err\n";
}

/* Verify the destination dtype is preserved (no silent upcast to float32). */
$d = new NDArray([[1, 2], [3, 4]], 'int64');
$d[0] = new NDArray([PHP_INT_MAX, PHP_INT_MIN], 'int64');
echo "int64-preserved: ",
     ($d[0][0] === PHP_INT_MAX && $d[0][1] === PHP_INT_MIN) ? "OK" : "BAD", "\n";

echo "done\n";
?>
--EXPECT--
float32: nda=OK src_intact=OK arr=OK mismatch=THROWN
float64: nda=OK src_intact=OK arr=OK mismatch=THROWN
int32: nda=OK src_intact=OK arr=OK mismatch=THROWN
int64: nda=OK src_intact=OK arr=OK mismatch=THROWN
int64-preserved: OK
done
