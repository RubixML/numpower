--TEST--
$a[i] = $scalar preserves precision for every dtype (no float32 round-trip)
--FILE--
<?php
/* offsetSet previously routed every scalar through float32 (via
   NDArray_CreateFromLongScalar), so PHP_INT_MAX for an int64 target became
   PHP_INT_MIN and 1234567890 for a uint64 target became 1234567936. The
   scalar fast path stays in the target's dtype end-to-end. */

$cases = [
    /* dtype     idx, value to set,           expected readback */
    ['int8',     1, 127,                       127],
    ['int8',     1, -128,                      -128],
    ['uint8',    1, 255,                       255],
    ['int16',    1, 32767,                     32767],
    ['int16',    1, -32768,                    -32768],
    ['uint16',   1, 65535,                     65535],
    ['int32',    1, 2147483647,                2147483647],
    ['int32',    1, -2147483648,               -2147483648],
    ['uint32',   1, 4294967295,                4294967295],
    ['int64',    1, PHP_INT_MAX,               PHP_INT_MAX],
    ['int64',    1, PHP_INT_MIN,               PHP_INT_MIN],
    ['float32',  1, 1.5,                       1.5],
    ['float64',  1, 1.5,                       1.5],
];

foreach ($cases as [$t, $idx, $val, $expected]) {
    $a = new NDArray([1, 2, 3], $t);
    $a[$idx] = $val;
    $got = $a[$idx];
    echo "$t [$idx]=$val: ", ($got === $expected ? "OK" : "BAD got=" . var_export($got, true)), "\n";
}

/* uint64 + string for max value */
$a = new NDArray(['0', '1', '2'], 'uint64');
$a[1] = '18446744073709551615';
echo "uint64 [1]='18446744073709551615': ", ($a[1] === '18446744073709551615' ? "OK" : "BAD"), "\n";

/* float128 + string for high precision */
$a = new NDArray(['0', '1', '2'], 'float128');
$a[1] = '3.14159265358979324';
echo "float128 [1]='3.14159265358979324': ", ($a[1] === '3.14159265358979324' ? "OK" : "BAD got=" . var_export($a[1], true)), "\n";
?>
--EXPECT--
int8 [1]=127: OK
int8 [1]=-128: OK
uint8 [1]=255: OK
int16 [1]=32767: OK
int16 [1]=-32768: OK
uint16 [1]=65535: OK
int32 [1]=2147483647: OK
int32 [1]=-2147483648: OK
uint32 [1]=4294967295: OK
int64 [1]=9223372036854775807: OK
int64 [1]=-9223372036854775808: OK
float32 [1]=1.5: OK
float64 [1]=1.5: OK
uint64 [1]='18446744073709551615': OK
float128 [1]='3.14159265358979324': OK
