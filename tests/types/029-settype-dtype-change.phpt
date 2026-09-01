--TEST--
NDArray::setType() mutates the array's data type in place (void)
--FILE--
<?php
/* setType() must:
     - mutate $this in place (same object identity after the call)
     - return void (null)
     - convert element values and PHP types to the target dtype
     - preserve the device (a CPU array stays CPU)
     - be a true no-op when the target dtype equals the current one
     - throw on an unknown dtype, leaving the array unchanged */

/* 1. float32 -> int32: values truncate, elements become ints, same object */
$a = new NDArray([1.5, 2.5, 3.7], 'float32');
$ret = $a->setType('int32');
$php = $a->toArray();
$ok = ($a instanceof NDArray) && $php === [1, 2, 3]
     && array_map('gettype', $php) === ['integer', 'integer', 'integer']
     && !$a->isGPU();
echo 'float32->int32: ', ($ok ? 'OK' : 'BAD'), ' ', json_encode($php), "\n";

/* 2. int32 -> float64: elements become floats, values preserved */
$b = new NDArray([1, 2, 3], 'int32');
$b->setType('float64');
$php = $b->toArray();
$ok = array_map('gettype', $php) === ['double', 'double', 'double']
     && $php === [1.0, 2.0, 3.0];
echo 'int32->float64: ', ($ok ? 'OK' : 'BAD'), ' ', json_encode($php), "\n";

/* 3. returns void (null), not a new array */
echo 'return-void: ', (($ret === null) ? 'OK' : 'BAD'),
     ' (', ($ret === null ? 'null' : 'non-null'), ')', "\n";

/* 4. unknown dtype throws and leaves the array unchanged */
$c = new NDArray([1.0, 2.0], 'float64');
$before = $c->toArray();
$ctypes = array_map('gettype', $before);
$threw = false;
$msg = '';
try { $c->setType('badtype'); }
catch (Throwable $t) { $threw = true; $msg = get_class($t) . ' | ' . $t->getMessage(); }
$still = $c->toArray();
$ok = $threw && $still === $before && array_map('gettype', $still) === $ctypes;
echo "badtype-throws-unchanged: ", ($ok ? 'OK' : 'BAD'), " ", $msg, "\n";

/* 5. same-dtype cast is a true no-op: values and types preserved */
$d = new NDArray([4, 5, 6], 'int32');
$d->setType('int32');
$php = $d->toArray();
$ok = $php === [4, 5, 6] && array_map('gettype', $php) === ['integer','integer','integer'] && !$d->isGPU();
echo 'same-dtype-noop: ', ($ok ? 'OK' : 'BAD'), ' ', json_encode($php), "\n";

/* 6. 0-D scalar (shape []) stays an NDArray; __toString is "7\n" */
$e = new NDArray(7.0);
$e->setType('int32');
$ok = ($e instanceof NDArray) && $e->shape() === [] && trim((string)$e) === '7';
echo 'zero-d: ', ($ok ? 'OK' : 'BAD'), ' shape=', json_encode($e->shape()), "\n";

/* 7. device preserved: a CPU array stays CPU after setType */
$f = new NDArray([1.0, 2.0, 3.0], 'float32');
$f->setType('float64');
$ok = ($f instanceof NDArray) && !$f->isGPU() && $f->toArray() === [1.0, 2.0, 3.0];
echo 'device-stays-CPU: ', ($ok ? 'OK' : 'BAD'), ' isGPU=', ($f->isGPU() ? 1 : 0), "\n";

/* 8. wrong argument count rejects */
try {
    (new NDArray([1, 2, 3], 'float32'))->setType();
    echo "no-arg: NO-THROW\n";
} catch (Throwable $t) {
    echo "no-arg threw: ", get_class($t), "\n";
}
?>
--EXPECT--
float32->int32: OK [1,2,3]
int32->float64: OK [1,2,3]
return-void: OK (null)
badtype-throws-unchanged: OK Error | Invalid data type 'badtype'. Supported: float4, float8, float16, float32, float64, float128, int8, uint8, int16, uint16, int32, uint32, int64, uint64
same-dtype-noop: OK [4,5,6]
zero-d: OK shape=[]
device-stays-CPU: OK isGPU=0
no-arg threw: ArgumentCountError
