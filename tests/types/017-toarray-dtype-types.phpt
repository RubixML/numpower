--TEST--
NDArray::toArray() returns the dtype-mandated PHP types for every dtype
--FILE--
<?php
/* toArray() must produce native PHP values whose types match the dtype:
     - float128, uint64                                 -> string
     - int8/16/32/64, uint8/16/32                       -> int
     - float4, float8, float16, float32, float64        -> float
   This guarantees PHP code can rely on is_int/is_float/is_string without
   manual casting. */

$cases = [
    'float4'   => [['1.5', '2', '-1.5'],  'float'],
    'float8'   => [['1.5', '2.5', '6'],   'float'],
    'float16'  => [['1.5', '2.5', '0.5'], 'float'],
    'float32'  => [[1.5, 2.5, 3.5],       'float'],
    'float64'  => [[1.5, 2.5, 3.5],       'float'],
    'float128' => [['1.5', '2.5', '3.5'], 'string'],

    'int8'     => [[-128, 0, 127],                                  'int'],
    'uint8'    => [[0, 128, 255],                                   'int'],
    'int16'    => [[-32768, 0, 32767],                              'int'],
    'uint16'   => [[0, 32768, 65535],                               'int'],
    'int32'    => [[-2147483648, 0, 2147483647],                    'int'],
    'uint32'   => [[0, 2147483648, 4294967295],                     'int'],
    'int64'    => [[PHP_INT_MIN, 0, PHP_INT_MAX],                   'int'],
    'uint64'   => [['0', '9223372036854775808', '18446744073709551615'], 'string'],
];

foreach ($cases as $dtype => [$values, $expected_type]) {
    $arr = new NDArray($values, $dtype);
    $php = $arr->toArray();
    $type_ok = true;
    foreach ($php as $v) {
        if ($expected_type === 'int'    && !is_int($v))    { $type_ok = false; break; }
        if ($expected_type === 'float'  && !is_float($v))  { $type_ok = false; break; }
        if ($expected_type === 'string' && !is_string($v)) { $type_ok = false; break; }
    }
    echo "$dtype: ", ($type_ok ? "OK ($expected_type)" : "BAD (expected $expected_type)"), "\n";
}
?>
--EXPECT--
float4: OK (float)
float8: OK (float)
float16: OK (float)
float32: OK (float)
float64: OK (float)
float128: OK (string)
int8: OK (int)
uint8: OK (int)
int16: OK (int)
uint16: OK (int)
int32: OK (int)
uint32: OK (int)
int64: OK (int)
uint64: OK (string)
