--TEST--
foreach over a 1-D NDArray yields elements with the dtype-mandated PHP type
--FILE--
<?php
/* foreach calls Iterator::current() which goes through the same scalar
   conversion path as $a[i]. The yielded values must therefore inherit the
   dtype's PHP-type contract (strings for float128/uint64, ints for other
   ints, floats for other floats). */

$expected = [
    'float4'   => 'float', 'float8'  => 'float', 'float16' => 'float',
    'float32'  => 'float', 'float64' => 'float',
    'float128' => 'string',
    'int8'     => 'int',   'uint8'   => 'int',
    'int16'    => 'int',   'uint16'  => 'int',
    'int32'    => 'int',   'uint32'  => 'int',
    'int64'    => 'int',
    'uint64'   => 'string',
];

foreach ($expected as $t => $want) {
    $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $vals = $strIO ? ['1','2','3'] : [1,2,3];
    $a = new NDArray($vals, $t);
    $ok = true;
    foreach ($a as $v) {
        if ($want === 'int'    && !is_int($v))    { $ok = false; break; }
        if ($want === 'float'  && !is_float($v))  { $ok = false; break; }
        if ($want === 'string' && !is_string($v)) { $ok = false; break; }
    }
    echo "$t: ", ($ok ? "OK ($want)" : "BAD (expected $want)"), "\n";
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
