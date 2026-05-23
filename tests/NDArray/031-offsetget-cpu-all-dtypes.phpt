--TEST--
NDArray::offsetGet() on CPU returns dtype-correct scalars for every dtype
--FILE--
<?php
/* offsetGet on a 1-D source produces a dtype-correct PHP scalar via
   NDArray_ScalarToZval:
     - string for float128 / uint64
     - int    for the other integer dtypes (int8..int64, uint8..uint32)
     - float  for the remaining floating-point dtypes (float4..float64)

   The same surface is exercised on GPU by tests/NDArray/gpu/011-* —
   this is the CPU counterpart, covering every supported dtype and including
   boundary values for each. */

function gettype_label($v) {
    if (is_int($v))    return 'int';
    if (is_float($v))  return 'float';
    if (is_string($v)) return 'string';
    return gettype($v);
}

$cases = [
    'float4'   => [['1.5', '-1.5', '2', '6'],                   'float'],
    'float8'   => [['0.5', '-0.5', '1.5', '6'],                 'float'],
    'float16'  => [['1.5', '-1.5', '0.5', '-0.5'],              'float'],
    'float32'  => [[0.0, -1.5, 1e3, -1e3],                      'float'],
    'float64'  => [[0.0, -1.5, 1e10, -1e10],                    'float'],
    'float128' => [['1000', '-1.5', '1.25', '1000000000'],      'string'],

    'int8'     => [[-128, -1, 0, 127],                          'int'],
    'uint8'    => [[0, 1, 128, 255],                            'int'],
    'int16'    => [[-32768, -1, 0, 32767],                      'int'],
    'uint16'   => [[0, 1, 32768, 65535],                        'int'],
    'int32'    => [[-2147483648, -1, 0, 2147483647],            'int'],
    'uint32'   => [[0, 1, 2147483648, 4294967295],              'int'],
    'int64'    => [[PHP_INT_MIN, -1, 0, PHP_INT_MAX],           'int'],
    'uint64'   => [['0', '1', '9223372036854775808', '18446744073709551615'], 'string'],
];

foreach ($cases as $dtype => [$values, $want_type]) {
    $a = new NDArray($values, $dtype);
    $type_ok = true;
    $val_ok  = true;
    foreach ($values as $i => $expected_input) {
        $v = $a[$i];
        if (gettype_label($v) !== $want_type) { $type_ok = false; }
        /* Each input was authored to round-trip exactly through the dtype
           (string for fp128/uint64, integer for ints, fp64-exact float for
           the remaining floats). So $a[$i] == $expected_input in PHP. */
        if ((string)$v !== (string)$expected_input) { $val_ok = false; }
    }
    echo "$dtype: ", ($type_ok ? 'type=OK' : 'type=BAD'), ' ',
                     ($val_ok ? 'value=OK' : 'value=BAD'), "\n";
}
?>
--EXPECT--
float4: type=OK value=OK
float8: type=OK value=OK
float16: type=OK value=OK
float32: type=OK value=OK
float64: type=OK value=OK
float128: type=OK value=OK
int8: type=OK value=OK
uint8: type=OK value=OK
int16: type=OK value=OK
uint16: type=OK value=OK
int32: type=OK value=OK
uint32: type=OK value=OK
int64: type=OK value=OK
uint64: type=OK value=OK
