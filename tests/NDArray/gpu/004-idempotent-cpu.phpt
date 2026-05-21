--TEST--
NDArray::cpu() on a CPU array (and back) preserves data for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* cpu() on a CPU-resident array must produce an independent CPU copy with
   identical contents. This exercises the NDArray_Copy CPU branch that, like
   the GPU branch, used to leave rtn->uuid uninitialised. */
$cases = [
    'float4'   => ['0', '0.5', '1', '1.5'],
    'float8'   => ['0', '0.5', '1', '1.5'],
    'float16'  => ['0', '0.5', '1', '1.5'],
    'float32'  => [0, 1, 0.5, 1.5],
    'float64'  => [0, 1, 0.5, 1.5],
    'float128' => ['0', '1', '0.5', '1.5'],
    'int8'     => [-128, -1, 0, 127],
    'uint8'    => [0, 1, 200, 255],
    'int16'    => [-32768, -1, 0, 32767],
    'uint16'   => [0, 1, 50000, 65535],
    'int32'    => [-2147483648, -1, 0, 2147483647],
    'uint32'   => [0, 1, 3000000000, 4294967295],
    'int64'    => [-1, 0, 1, 9223372036854775807],
    'uint64'   => ['0', '1', '9223372036854775807', '18446744073709551615'],
];

foreach ($cases as $dtype => $values) {
    $a = new NDArray($values, $dtype);
    $b = $a->cpu();
    $c = $b->cpu();
    echo $dtype, ': ',
         (!$a->isGPU() && !$b->isGPU() && !$c->isGPU() &&
          (string)$a === (string)$b && (string)$b === (string)$c ? 'OK' : 'FAIL'),
         "\n";
}
?>
--EXPECT--
float4: OK
float8: OK
float16: OK
float32: OK
float64: OK
float128: OK
int8: OK
uint8: OK
int16: OK
uint16: OK
int32: OK
uint32: OK
int64: OK
uint64: OK
