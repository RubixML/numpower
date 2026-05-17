--TEST--
toArray() after CPU -> GPU -> CPU yields dtype-mandated PHP types unchanged
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Move every dtype to GPU and back, then toArray(). The PHP element types
   must still match dtype expectations: strings for float128 & uint64,
   ints for the rest of int/uint, floats for the rest of float. */

$cases = [
    'float4'   => [['1.5', '-1.5', '2', '6'],     'float'],
    'float8'   => [['0.5', '-0.5', '1.5', '6'],   'float'],
    'float16'  => [['1.5', '-1.5', '0.5', '-0.5'],'float'],
    'float32'  => [[1.0, 2.0, 3.0],               'float'],
    'float64'  => [[1.0, 2.0, 3.0],               'float'],
    'float128' => [['1.5', '-1.5', '1e-9'],       'string'],
    'int8'     => [[-128, 0, 127],                'int'],
    'uint8'    => [[0, 128, 255],                 'int'],
    'int16'    => [[-32768, 0, 32767],            'int'],
    'uint16'   => [[0, 32768, 65535],             'int'],
    'int32'    => [[-2147483648, 0, 2147483647],  'int'],
    'uint32'   => [[0, 2147483648, 4294967295],   'int'],
    'int64'    => [[PHP_INT_MIN, 0, PHP_INT_MAX], 'int'],
    'uint64'   => [['0', '1', '18446744073709551615'], 'string'],
];

foreach ($cases as $dtype => [$values, $expected_type]) {
    $arr  = new NDArray($values, $dtype);
    $gpu  = $arr->gpu();
    $back = $gpu->cpu();
    $php  = $back->toArray();
    $ok   = true;
    foreach ($php as $v) {
        if ($expected_type === 'int'    && !is_int($v))    { $ok = false; break; }
        if ($expected_type === 'float'  && !is_float($v))  { $ok = false; break; }
        if ($expected_type === 'string' && !is_string($v)) { $ok = false; break; }
    }
    echo "$dtype: ", ($ok ? "OK ($expected_type)" : "BAD"), "\n";
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
