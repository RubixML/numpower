--TEST--
NDArray::astype() returns a new NDArray with the target dtype, preserving device
--FILE--
<?php
/* astype() must:
     - return a NEW NDArray (original unchanged)
     - convert element values and PHP types to the target dtype
     - throw on an unknown dtype
     - be a no-op copy when the target equals the source dtype */

/* 1. float32 -> int32: values truncate, elements become ints.
   Capture the original's values BEFORE the cast for an "unchanged" check. */
$a = new NDArray([1.5, 2.5, 3.7], 'float32');
$original_vals = $a->toArray();
$r = $a->astype('int32');
$php = $r->toArray();
$ok = $r instanceof NDArray && $php === [1, 2, 3]
     && array_map('gettype', $php) === ['integer', 'integer', 'integer'];
echo 'float32->int32: ', ($ok ? 'OK' : 'BAD'), ' ', json_encode($php), "\n";

/* 2. int32 -> float64: elements become floats, values preserved */
$b = new NDArray([1, 2, 3], 'int32');
$r = $b->astype('float64');
$php = $r->toArray();
$ok = $r instanceof NDArray && $php === [1.0, 2.0, 3.0]
     && array_map('gettype', $php) === ['double', 'double', 'double'];
echo 'int32->float64: ', ($ok ? 'OK' : 'BAD'), ' ', json_encode($php), "\n";

/* 3. original array is unchanged by the cast (still the same float32 values) */
$ok = $a->toArray() === $original_vals;
echo 'original-unchanged: ', ($ok ? 'OK' : 'BAD'), ' ', json_encode($a->toArray()), "\n";

/* 4. device preserved: a CPU array stays CPU after astype */
$c = new NDArray([1.0, 2.0, 3.0], 'float32');
$rc = $c->astype('float64');
$ok = $rc instanceof NDArray && !$rc->isGPU() && $rc->toArray() === [1.0, 2.0, 3.0];
echo 'device-stays-CPU: ', ($ok ? 'OK' : 'BAD'), ' isGPU=', ($rc->isGPU() ? 1 : 0), "\n";

/* 5. same-dtype cast returns a new, equal array */
$d = new NDArray([4, 5, 6], 'int32');
$rd = $d->astype('int32');
$ok = ($rd !== $d) && $rd->toArray() === [4, 5, 6] && !$rd->isGPU();
echo 'same-dtype-copy: ', ($ok ? 'OK' : 'BAD'), "\n";

/* 6. 0-D scalar (shape []) stays an NDArray after astype; __toString is "7\n" */
$e = new NDArray(7.0);
$re = $e->astype('int32');
$ok = $re instanceof NDArray && $re->shape() === [] && trim((string)$re) === '7';
echo 'zero-d: ', ($ok ? 'OK' : 'BAD'), ' shape=', json_encode($re->shape()), "\n";

/* 7. unknown dtype throws, message matches the canonical list */
try {
    (new NDArray([1, 2, 3], 'float32'))->astype('badtype');
    echo "badtype: NO-THROW\n";
} catch (Throwable $t) {
    echo "badtype threw: ", get_class($t), " | ", $t->getMessage(), "\n";
}

/* 8. wrong argument count rejects */
try {
    (new NDArray([1, 2, 3], 'float32'))->astype();
    echo "no-arg: NO-THROW\n";
} catch (Throwable $t) {
    echo "no-arg threw: ", get_class($t), "\n";
}
?>
--EXPECT--
float32->int32: OK [1,2,3]
int32->float64: OK [1,2,3]
original-unchanged: OK [1.5,2.5,3.700000047683716]
device-stays-CPU: OK isGPU=0
same-dtype-copy: OK
zero-d: OK shape=[]
badtype threw: Error | Invalid data type 'badtype'. Supported: float4, float8, float16, float32, float64, float128, int8, uint8, int16, uint16, int32, uint32, int64, uint64
no-arg threw: ArgumentCountError
