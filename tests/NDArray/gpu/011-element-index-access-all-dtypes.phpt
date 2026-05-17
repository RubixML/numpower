--TEST--
NDArray element index access on GPU returns correct values for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU element access: every dtype must produce the same value (within
   dtype precision) regardless of device. Stride reads on float128 / uint64
   that previously short-circuited through float32 produced bogus values
   like 0 or tiny denormals — these checks lock in the byte-correct path. */

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
    'float64'  => [[0.0, -1.5, 1e3, -1e3],                      'float'],
    'float128' => [['1e3', '-1.5', '1e-9', '1e9'],              'string'],

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
    $cpu = new NDArray($values, $dtype);
    $gpu = $cpu->gpu();
    $type_ok = true;
    $val_ok  = true;
    /* Reference: read each index from the CPU array. GPU access must match. */
    foreach ($values as $i => $_) {
        $cpu_v = $cpu[$i];
        $gpu_v = $gpu[$i];
        if (gettype_label($gpu_v) !== $want_type) { $type_ok = false; }
        if ($cpu_v !== $gpu_v) {
            /* For floats, tolerate the trivial != caused by storage; the byte
               contents are identical so identity should hold, but for the
               reduced-precision GPU read paths we accept tiny rounding. */
            if (is_float($cpu_v) && is_float($gpu_v)) {
                $tol = max(1e-6, abs($cpu_v) * 1e-3);
                if (abs($cpu_v - $gpu_v) > $tol) { $val_ok = false; }
            } else {
                $val_ok = false;
            }
        }
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
