--TEST--
NDArray::setDataType() mutates dtype in place; dataType() returns current type
--FILE--
<?php
/* setDataType() must:
     - mutate $this in place (no change to object identity)
     - return void (null)
     - convert element values and PHP types to the target dtype
     - preserve the device (a CPU array stays CPU)
     - be a no-op when the target dtype equals the current one
     - throw on an unknown dtype, leaving the array and its dataType() unchanged

   dataType() must:
     - return the current dtype as a string (e.g. 'float32')
     - reflect the immediately-preceding setDataType() call */

/* 1. float32 -> int32: values truncate, elements become ints;
      in-place: same PHP object identity, alias observes the change */
$a = new NDArray([1.5, 2.5, 3.7], 'float32');
$aliasA = $a;                                /* alias must share the object */
$idA = spl_object_id($a);
$pre = $a->dataType();
$ret = $a->setDataType('int32');
$php = $a->toArray();
$post = $a->dataType();
$aliasOK = (spl_object_id($aliasA) === $idA)
    && ($aliasA->dataType() === 'int32')
    && ($aliasA->toArray() === [1, 2, 3])
    && (array_map('gettype', $aliasA->toArray()) === ['integer','integer','integer']);
$ok = ($pre === 'float32') && $post === 'int32'
     && (spl_object_id($a) === $idA)         /* same PHP object after setDataType */
     && ($ret === null) && $php === [1, 2, 3]
     && array_map('gettype', $php) === ['integer', 'integer', 'integer']
     && !$a->isGPU()
     && $aliasOK;
echo "float32->int32: pre=$pre post=$post ret=", ($ret === null ? 'null' : 'non-null'),
     " values=", json_encode($php),
     " ok=", ($ok ? 'OK' : 'BAD'), "\n";

/* 2. int32 -> float64: elements become PHP floats (values preserved) */
$b = new NDArray([1, 2, 3], 'int32');
$before = $b->dataType();
$b->setDataType('float64');
$after = $b->dataType();
$php = $b->toArray();
$ok = $before === 'int32' && $after === 'float64'
     && array_map('gettype', $php) === ['double', 'double', 'double']
     && $php === [1.0, 2.0, 3.0];
echo "int32->float64: pre=$before post=$after",
     " values=", json_encode($php),
     " types=", json_encode(array_map('gettype', $php)),
     " ok=", ($ok ? 'OK' : 'BAD'), "\n";

/* 3. unknown dtype throws; dataType() and values unchanged */
$c = new NDArray([1.0, 2.0], 'float64');
$idC = spl_object_id($c);
$beforeType = $c->dataType();
$beforeVals = $c->toArray();
$threw = false; $tclass = '';
try { $c->setDataType('badtype'); }
catch (Throwable $t) { $threw = true; $tclass = get_class($t); }
$ok = $threw
     && (spl_object_id($c) === $idC)         /* object NOT replaced on throw */
     && $c->dataType() === $beforeType
     && $c->toArray() === $beforeVals
     && array_map('gettype', $c->toArray()) === ['double', 'double'];
echo "badtype: threw=", $tclass, " pre=$beforeType post=", $c->dataType(),
     " vals=", json_encode($c->toArray()),
     " ok=", ($ok ? 'OK' : 'BAD'), "\n";

/* 4. same-dtype cast is a true no-op: dtype and values unchanged */
$d = new NDArray([4, 5, 6], 'int32');
$idD = spl_object_id($d);
$d->setDataType('int32');
$ok = (spl_object_id($d) === $idD)           /* no-op must not replace object */
     && $d->dataType() === 'int32'
     && $d->toArray() === [4, 5, 6]
     && array_map('gettype', $d->toArray()) === ['integer','integer','integer']
     && !$d->isGPU();
echo "no-op: type=", $d->dataType(), " vals=", json_encode($d->toArray()),
     " ok=", ($ok ? 'OK' : 'BAD'), "\n";

/* 5. 0-D scalar (shape []) stays an NDArray; __toString is "7\n" */
$e = new NDArray(7.0, 'float64');
$pre = $e->dataType();
$e->setDataType('int32');
$post = $e->dataType();
$ok = ($e instanceof NDArray) && $e->shape() === []
     && ($pre === 'float64') && ($post === 'int32')
     && trim((string)$e) === '7';
echo "0-d: shape=", json_encode($e->shape()), " pre=$pre post=$post",
     " str=", json_encode((string)$e), " ok=", ($ok ? 'OK' : 'BAD'), "\n";

/* 6. device preserved: a CPU array stays CPU after setDataType */
$f = new NDArray([1.0, 2.0, 3.0], 'float64');
$f->setDataType('float32');
$ok = ($f instanceof NDArray) && !$f->isGPU()
     && $f->toArray() === [1.0, 2.0, 3.0]
     && $f->dataType() === 'float32';
echo "cpu: isGPU=", ($f->isGPU() ? 1 : 0), " type=", $f->dataType(),
     " vals=", json_encode($f->toArray()), " ok=", ($ok ? 'OK' : 'BAD'), "\n";

/* 7. wrong argument counts reject */
try {
    (new NDArray([1, 2, 3], 'float32'))->setDataType();
    echo "no-arg setDataType: NO-THROW\n";
} catch (Throwable $t) {
    echo "no-arg setDataType threw: ", get_class($t), "\n";
}
try {
    (new NDArray([1, 2, 3], 'float32'))->dataType(123);
    echo "dataType(123): NO-THROW\n";
} catch (Throwable $t) {
    echo "dataType(123) threw: ", get_class($t), "\n";
}
?>
--EXPECT--
float32->int32: pre=float32 post=int32 ret=null values=[1,2,3] ok=OK
int32->float64: pre=int32 post=float64 values=[1,2,3] types=["double","double","double"] ok=OK
badtype: threw=Error pre=float64 post=float64 vals=[1,2] ok=OK
no-op: type=int32 vals=[4,5,6] ok=OK
0-d: shape=[] pre=float64 post=int32 str="7\n" ok=OK
cpu: isGPU=0 type=float32 vals=[1,2,3] ok=OK
no-arg setDataType threw: ArgumentCountError
dataType(123) threw: ArgumentCountError
