--TEST--
NDArray::gpu() is idempotent for every dtype (regression for NDArray_Copy bug)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Calling gpu() on an array that is already on the GPU used to crash for
   every dtype except float32/float64 because NDArray_Copy did not initialise
   rtn->uuid and did not invoke cudaMemcpy for sub-byte / integer types.
   The chain CPU -> GPU -> GPU(copy) -> CPU must reproduce the original. */
$cases = [
    'float4'   => ['0', '0.5', '1', '1.5'],
    'float8'   => ['0', '0.5', '1', '1.5'],
    'float16'  => ['0', '0.5', '1', '1.5'],
    'float32'  => [0, 1, 0.5, 1.5],
    'float64'  => [0, 1, 0.5, 1.5],
    'float128' => ['0', '1', '0.5', '1.5'],
    'int8'     => [-1, 0, 1, 127],
    'uint8'    => [0, 1, 127, 255],
    'int16'    => [-1, 0, 1, 32767],
    'uint16'   => [0, 1, 32767, 65535],
    'int32'    => [-1, 0, 1, 2147483647],
    'uint32'   => [0, 1, 2147483647, 4294967295],
    'int64'    => [-1, 0, 1, 9223372036854775807],
    'uint64'   => ['0', '1', '9223372036854775807', '18446744073709551615'],
];

foreach ($cases as $dtype => $values) {
    $a = (new NDArray($values, $dtype))->gpu();
    $b = $a->gpu();
    $c = $b->gpu();
    $back = $c->cpu();
    $ref  = (string)(new NDArray($values, $dtype));
    echo $dtype, ': ',
         ($a->isGPU() && $b->isGPU() && $c->isGPU() && !$back->isGPU() &&
          (string)$back === $ref ? 'OK' : 'FAIL'),
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
