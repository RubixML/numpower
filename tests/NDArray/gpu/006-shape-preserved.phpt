--TEST--
NDArray::gpu()/cpu() preserves shape across the round-trip for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Multi-dimensional inputs (1D … 4D) must come back with identical shape on
   every supported dtype. */
$inputs = [
    [[4],          [1, 2, 3, 4]],
    [[2, 3],       [[1, 2, 3], [4, 5, 6]]],
    [[2, 2, 2],    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]],
    [[2, 1, 3, 2], [[[[1, 2], [3, 4], [5, 6]]],
                    [[[7, 8], [9, 10], [11, 12]]]]],
];

$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];

foreach ($inputs as [$shape, $payload]) {
    foreach ($dtypes as $dtype) {
        $a = new NDArray($payload, $dtype);
        $r = $a->gpu()->gpu()->cpu()->cpu();
        $shape_ok  = $r->shape() === $a->shape();
        $values_ok = (string)$r === (string)$a;
        echo $dtype, ' shape=[', implode(',', $a->shape()), ']: ',
             ($shape_ok && $values_ok ? 'OK' : 'FAIL'), "\n";
    }
}
?>
--EXPECT--
float4 shape=[4]: OK
float8 shape=[4]: OK
float16 shape=[4]: OK
float32 shape=[4]: OK
float64 shape=[4]: OK
float128 shape=[4]: OK
int8 shape=[4]: OK
uint8 shape=[4]: OK
int16 shape=[4]: OK
uint16 shape=[4]: OK
int32 shape=[4]: OK
uint32 shape=[4]: OK
int64 shape=[4]: OK
uint64 shape=[4]: OK
float4 shape=[2,3]: OK
float8 shape=[2,3]: OK
float16 shape=[2,3]: OK
float32 shape=[2,3]: OK
float64 shape=[2,3]: OK
float128 shape=[2,3]: OK
int8 shape=[2,3]: OK
uint8 shape=[2,3]: OK
int16 shape=[2,3]: OK
uint16 shape=[2,3]: OK
int32 shape=[2,3]: OK
uint32 shape=[2,3]: OK
int64 shape=[2,3]: OK
uint64 shape=[2,3]: OK
float4 shape=[2,2,2]: OK
float8 shape=[2,2,2]: OK
float16 shape=[2,2,2]: OK
float32 shape=[2,2,2]: OK
float64 shape=[2,2,2]: OK
float128 shape=[2,2,2]: OK
int8 shape=[2,2,2]: OK
uint8 shape=[2,2,2]: OK
int16 shape=[2,2,2]: OK
uint16 shape=[2,2,2]: OK
int32 shape=[2,2,2]: OK
uint32 shape=[2,2,2]: OK
int64 shape=[2,2,2]: OK
uint64 shape=[2,2,2]: OK
float4 shape=[2,1,3,2]: OK
float8 shape=[2,1,3,2]: OK
float16 shape=[2,1,3,2]: OK
float32 shape=[2,1,3,2]: OK
float64 shape=[2,1,3,2]: OK
float128 shape=[2,1,3,2]: OK
int8 shape=[2,1,3,2]: OK
uint8 shape=[2,1,3,2]: OK
int16 shape=[2,1,3,2]: OK
uint16 shape=[2,1,3,2]: OK
int32 shape=[2,1,3,2]: OK
uint32 shape=[2,1,3,2]: OK
int64 shape=[2,1,3,2]: OK
uint64 shape=[2,1,3,2]: OK
