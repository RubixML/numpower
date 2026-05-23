--TEST--
NDArray Iterator: 0-D arrays reject $a[i] / $a[i] = v / isset($a[i]) (no segfault, no leak)
--FILE--
<?php
/* The scalar factory paths (_createScalarFromDouble / _createScalarFromString
   and the four legacy _create{Float32,Double64}From{Long,Double}Scalar helpers)
   skip NDArrayIterator_INIT, leaving the iterator pointers NULL. The Array
   Access methods used to read NDArray_SHAPE(ndarray)[0] and dereference
   ndarray->iterator without checking ndim, which segfaulted on 0-D inputs.

   We assert that all four ArrayAccess operations now fail cleanly on every
   dtype that goes through a scalar factory. */

$dtypes = ['float32', 'float64', 'float128', 'int8', 'uint8', 'int16',
           'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($dtypes as $t) {
    $str_io = in_array($t, ['float128','int64','uint64'], true);
    $val = $str_io ? '7' : 7;
    $s = new NDArray($val, $t);

    /* isset returns false, doesn't throw. */
    $exists = isset($s[0]) ? '1' : '0';

    /* offsetGet throws. */
    $get_err = 'NONE';
    try { $x = $s[0]; }
    catch (Throwable $e) { $get_err = $e->getMessage(); }

    /* offsetSet throws. */
    $set_err = 'NONE';
    try { $s[0] = 1; }
    catch (Throwable $e) { $set_err = $e->getMessage(); }

    echo "$t: isset=$exists get='$get_err' set='$set_err'\n";
    unset($s);
}

echo "done\n";
?>
--EXPECT--
float32: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
float64: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
float128: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
int8: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
uint8: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
int16: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
uint16: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
int32: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
uint32: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
int64: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
uint64: isset=0 get='Cannot index a 0-D NDArray (no axis 0)' set='Cannot index a 0-D NDArray (no axis 0)'
done
