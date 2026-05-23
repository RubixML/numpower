--TEST--
NDArray::dump() on 0-D scalars does not segfault for any dtype
--FILE--
<?php
/* The dtype-specific scalar factories (`_createScalarFromDouble`,
   `_createScalarFromString`, etc. in src/ndarray/frontend/ndarray_factory.c)
   build 0-D NDArrays with iterator == NULL because there is no axis to
   iterate. NDArray_Dump (and NDArrayIterator_DUMP) used to dereference
   that field unconditionally — calling `(new NDArray(1, 'float128'))->dump()`
   would segfault.

   The fix in src/debug.c guards both prints with an explicit `if
   (a->iterator != NULL)`. Verifies every supported dtype survives. */
$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];
foreach ($dtypes as $t) {
    $s = new NDArray(1, $t);
    ob_start();
    $s->dump();
    $out = ob_get_clean();
    /* Every dump must contain the dtype name (basic sanity) and the
       "(none)" iterator marker (0-D scalar has no iterator). */
    if (strpos($out, $t) === false || strpos($out, '(none)') === false) {
        echo "$t: FAIL\n$out\n";
        exit;
    }
}
echo "ok\n";
?>
--EXPECT--
ok
