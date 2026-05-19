--TEST--
NDArray element index access ($a[i]) returns correct values for every dtype on CPU
--FILE--
<?php
/* For every supported dtype, build a 1-D NDArray and read every element via
   $a[i]. We verify both the value and its native PHP type:
     - float128, uint64                                 -> string
     - int8/16/32/64, uint8/16/32                       -> int
     - float4, float8, float16, float32, float64        -> float
   Boundary values are chosen so that an incorrect dtype reinterpretation
   would yield a measurably different result. */

function gettype_label($v) {
    if (is_int($v))    return 'int';
    if (is_float($v))  return 'float';
    if (is_string($v)) return 'string';
    return gettype($v);
}

$cases = [
    /* dtype     => [values,                                    expected per index] */
    'float4'   => [['1.5', '-1.5', '2', '6'],                   ['1.5', '-1.5', '2', '6']],
    'float8'   => [['0.5', '-0.5', '1.5', '6'],                 ['0.5', '-0.5', '1.5', '6']],
    'float16'  => [['1.5', '-1.5', '0.5', '-0.5'],              ['1.5', '-1.5', '0.5', '-0.5']],
    'float32'  => [[0.0, -1.5, 1e3, -1e3],                      ['0', '-1.5', '1000', '-1000']],
    'float64'  => [[0.0, -1.5, 1e3, -1e3],                      ['0', '-1.5', '1000', '-1000']],
    /* float128 stringification uses libquadmath's %Qg with 34 significant
       digits, so values not exactly representable in 113 bits (like 1e-9)
       expand to the closest representable approximation. */
    'float128' => [['1e3', '-1.5', '1e-9', '1e9'],              ['1000', '-1.5', '9.999999999999999999999999999999999e-10', '1000000000']],

    'int8'     => [[-128, -1, 0, 127],                          ['-128', '-1', '0', '127']],
    'uint8'    => [[0, 1, 128, 255],                            ['0', '1', '128', '255']],
    'int16'    => [[-32768, -1, 0, 32767],                      ['-32768', '-1', '0', '32767']],
    'uint16'   => [[0, 1, 32768, 65535],                        ['0', '1', '32768', '65535']],
    'int32'    => [[-2147483648, -1, 0, 2147483647],            ['-2147483648', '-1', '0', '2147483647']],
    'uint32'   => [[0, 1, 2147483648, 4294967295],              ['0', '1', '2147483648', '4294967295']],
    'int64'    => [[PHP_INT_MIN, -1, 0, PHP_INT_MAX],           ['-9223372036854775808', '-1', '0', '9223372036854775807']],
    'uint64'   => [['0', '1', '9223372036854775808', '18446744073709551615'],
                    ['0', '1', '9223372036854775808', '18446744073709551615']],
];

/* Expected PHP type per dtype */
$expected_type = [
    'float4'   => 'float', 'float8'  => 'float', 'float16' => 'float',
    'float32'  => 'float', 'float64' => 'float',
    'float128' => 'string',
    'int8'     => 'int',   'uint8'   => 'int',
    'int16'    => 'int',   'uint16'  => 'int',
    'int32'    => 'int',   'uint32'  => 'int',
    'int64'    => 'int',
    'uint64'   => 'string',
];

foreach ($cases as $dtype => [$values, $expected_str]) {
    $arr = new NDArray($values, $dtype);
    $want_type = $expected_type[$dtype];
    $ok_value = true;
    $ok_type  = true;
    foreach ($values as $i => $_) {
        $v = $arr[$i];
        if (gettype_label($v) !== $want_type) {
            $ok_type = false;
        }
        $exp = $expected_str[$i];
        if ($want_type === 'float') {
            /* Compare numerically: tolerate dtype-specific rounding. */
            $delta = abs((float)$v - (float)$exp);
            $tol   = max(1e-6, abs((float)$exp) * 1e-3);
            if ($delta > $tol) { $ok_value = false; }
        } else {
            /* int / string compare via string form. */
            if ((string)$v !== $exp) { $ok_value = false; }
        }
    }
    echo "$dtype: ", ($ok_type ? 'type=OK' : 'type=BAD'), ' ',
                     ($ok_value ? 'value=OK' : 'value=BAD'), "\n";
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
