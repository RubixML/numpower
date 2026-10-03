--TEST--
NDArray::cast() / toDataType() change the dtype in place and return $this
--FILE--
<?php
/* 1. Ticket example: a bare call changes $ndarray itself */
$a = new NDArray([1.5, 2.5, 3.7], 'float32');
$alias = $a;
$id = spl_object_id($a);
$a->cast('int32');
echo $a->dataType(), ' ', json_encode($a->toArray()), "\n";
echo spl_object_id($a) === $id ? "same object\n" : "NEW OBJECT\n";
echo $alias->dataType(), " (alias)\n";

/* 2. toDataType() alias; return value is $this */
$ret = $a->toDataType('float64');
echo $ret === $a ? "returns \$this\n" : "returns other\n";
echo $a->dataType(), ' ', json_encode($a->toArray()), "\n";

/* 3. Ticket example with float16; chaining */
$h = new NDArray([1.5, 2.5, 3.7], 'float32');
echo json_encode($h->cast('float16')->toArray()), ' ', $h->dataType(), "\n";

/* 4. Same dtype: no-op, still returns $this */
echo $h->cast('float16') === $h ? "same-type ok\n" : "same-type BAD\n";

/* 5. Keep the original by casting a copy */
$src = new NDArray([1.5, 2.5], 'float32');
$dst = NumPower::copy($src)->cast('int8');
echo $src->dataType(), ' ', json_encode($src->toArray()), ' | ',
     $dst->dataType(), ' ', json_encode($dst->toArray()), "\n";

/* 6. Invalid dtype throws, array unchanged */
try {
    $src->cast('bogus');
} catch (Error $e) {
    echo "Error caught\n";
}
echo $src->dataType(), "\n";
?>
--EXPECT--
int32 [1,2,3]
same object
int32 (alias)
returns $this
float64 [1,2,3]
[1.5,2.5,3.69921875] float16
same-type ok
float32 [1.5,2.5] | int8 [1,2]
Error caught
float32
