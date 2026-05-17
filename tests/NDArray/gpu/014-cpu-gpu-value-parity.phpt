--TEST--
CPU and GPU element access return identical values at boundary inputs for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* For every dtype, build a 1-D array containing min/max representative
   values plus zero, then verify that $cpu[i] === $gpu[i] (or is_nan parity)
   for every index. A failure here means the GPU memcpy / reinterpretation
   path diverges from the CPU read for some dtype. */

$cases = [
    'int8'     => [-128, -1, 0, 1, 127],
    'uint8'    => [0, 1, 127, 128, 255],
    'int16'    => [-32768, -1, 0, 1, 32767],
    'uint16'   => [0, 1, 32767, 32768, 65535],
    'int32'    => [-2147483648, -1, 0, 1, 2147483647],
    'uint32'   => [0, 1, 2147483647, 2147483648, 4294967295],
    'int64'    => [PHP_INT_MIN, -1, 0, 1, PHP_INT_MAX],
    'uint64'   => ['0', '1', '9223372036854775807', '9223372036854775808', '18446744073709551615'],
    'float4'   => ['-6', '-0.5', '0', '0.5', '6'],
    'float8'   => ['-240', '-0.5', '0', '0.5', '240'],
    'float16'  => ['-65504', '-1.5', '0', '1.5', '65504'],
    'float32'  => [-1e38, -1.5, 0.0, 1.5, 1e38],
    'float64'  => [-1e308, -1.5, 0.0, 1.5, 1e308],
    'float128' => ['-1e4000', '-1.5', '0', '1.5', '1e4000'],
];

foreach ($cases as $dtype => $vals) {
    $cpu = new NDArray($vals, $dtype);
    $gpu = $cpu->gpu();
    $ok = true;
    foreach ($vals as $i => $_) {
        $cv = $cpu[$i];
        $gv = $gpu[$i];
        if (is_float($cv) && is_nan($cv) && is_nan($gv)) continue;
        if ($cv !== $gv) { $ok = false; break; }
    }
    echo "$dtype: ", ($ok ? "OK" : "BAD"), "\n";
}
?>
--EXPECT--
int8: OK
uint8: OK
int16: OK
uint16: OK
int32: OK
uint32: OK
int64: OK
uint64: OK
float4: OK
float8: OK
float16: OK
float32: OK
float64: OK
float128: OK
