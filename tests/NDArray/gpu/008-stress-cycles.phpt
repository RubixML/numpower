--TEST--
NDArray::gpu()/cpu() stress: many transfer cycles preserve data and don't crash
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Run a long sequence of CPU↔GPU transfers per dtype. Earlier the GPU branch
   of NDArray_Copy left rtn->uuid uninitialised, so the very second hop into
   GPU memory could land on a dangling buffer slot. A long bounce loop is the
   simplest way to make any such regression manifest. */
$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];

foreach ($dtypes as $dtype) {
    $a = new NDArray([0, 1, 2, 3], $dtype);
    $ref = (string)$a;
    $cur = $a;
    for ($i = 0; $i < 32; $i++) {
        $cur = ($i % 2 === 0) ? $cur->gpu() : $cur->cpu();
    }
    $final = $cur->cpu();
    echo $dtype, ': ', ((string)$final === $ref ? 'OK' : 'FAIL'), "\n";
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
