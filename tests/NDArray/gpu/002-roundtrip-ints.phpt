--TEST--
NDArray::gpu()/cpu() preserves data for all integer dtypes (with boundary values)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Each integer dtype is exercised with values that span its full representable
   range — boundaries included — so any silent narrowing during the host↔device
   transfer is caught. uint64 must use string input to avoid PHP_INT overflow. */
$cases = [
    'int8'   => [-128, -1, 0, 1, 127],
    'uint8'  => [0, 1, 127, 200, 255],
    'int16'  => [-32768, -1, 0, 1, 32767],
    'uint16' => [0, 1, 32767, 50000, 65535],
    'int32'  => [-2147483648, -1, 0, 1, 2147483647],
    'uint32' => [0, 1, 2147483647, 3000000000, 4294967295],
    'int64'  => [-9223372036854775807-1, -1, 0, 1, 9223372036854775807],
    'uint64' => ['0', '1', '9223372036854775807', '18446744073709551615'],
];

foreach ($cases as $dtype => $values) {
    $cpu = new NDArray($values, $dtype);
    $before = (string)$cpu;
    $gpu = $cpu->gpu();
    if (!$gpu->isGPU()) {
        echo "$dtype: gpu() did not move to GPU\n";
        continue;
    }
    $back = $gpu->cpu();
    if ($back->isGPU()) {
        echo "$dtype: cpu() did not move to CPU\n";
        continue;
    }
    $after = (string)$back;
    echo $dtype, ': ', ($before === $after ? 'OK' : "DIFF before=$before after=$after"), "\n";
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
