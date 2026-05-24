--TEST--
NumPower::ones($shape, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* ones() must accept every dtype the engine knows about and produce a
   buffer whose elements (a) decode as 1 and (b) round-trip through the
   dtype-mandated PHP type. The element-type check guards against accidental
   float32 dispatch when a different dtype was requested — which is exactly
   the dormant bug the refactor closes (the old NDArray_Ones always wrote
   float32 1.0s, half-trampling the buffer on wider dtypes). */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

/* Expected PHP type returned by `$a[i]` per dtype (mirrors
   tests/types/025-foreach-dtype-preservation.phpt). */
$expected_type = [
    'float4'   => 'double',  'float8'   => 'double',  'float16'  => 'double',
    'float32'  => 'double',  'float64'  => 'double',  'float128' => 'string',
    'int8'     => 'integer', 'uint8'    => 'integer',
    'int16'    => 'integer', 'uint16'   => 'integer',
    'int32'    => 'integer', 'uint32'   => 'integer',
    'int64'    => 'integer', 'uint64'   => 'string',
];

foreach ($dtypes as $dt) {
    $a = NumPower::ones([3], $dt);
    $v = $a[0];
    $is_one = ($v === 1 || $v === 1.0 || $v === '1' || $v === '1.0');
    echo $dt, ': type=', gettype($v),
         ' expect=', $expected_type[$dt],
         ' val_one=', ($is_one ? 'OK' : 'BAD'),
         ' device_cpu=', ($a->isGPU() ? 'BAD' : 'OK'),
         "\n";
}

/* The sum of `n` ones must equal `n` for every dtype that can represent
   small ints exactly. fp4 can only express {0, 0.5, 1, 1.5, ...} so
   1+1+...+1 (8 times) overflows the magnitude range; check sum within
   the dtype's representable range. */
$ns = ['float4' => 4, 'float8' => 8, 'float16' => 8, 'float32' => 8,
       'float64' => 8, 'float128' => 8, 'int8' => 8, 'uint8' => 8,
       'int16' => 8, 'uint16' => 8, 'int32' => 8, 'uint32' => 8,
       'int64' => 8, 'uint64' => 8];
foreach ($dtypes as $dt) {
    $n = $ns[$dt];
    $a = NumPower::ones([$n], $dt);
    echo $dt, ' sum_of_', $n, '=', (string)NumPower::sum($a), "\n";
}
?>
--EXPECT--
float4: type=double expect=double val_one=OK device_cpu=OK
float8: type=double expect=double val_one=OK device_cpu=OK
float16: type=double expect=double val_one=OK device_cpu=OK
float32: type=double expect=double val_one=OK device_cpu=OK
float64: type=double expect=double val_one=OK device_cpu=OK
float128: type=string expect=string val_one=OK device_cpu=OK
int8: type=integer expect=integer val_one=OK device_cpu=OK
uint8: type=integer expect=integer val_one=OK device_cpu=OK
int16: type=integer expect=integer val_one=OK device_cpu=OK
uint16: type=integer expect=integer val_one=OK device_cpu=OK
int32: type=integer expect=integer val_one=OK device_cpu=OK
uint32: type=integer expect=integer val_one=OK device_cpu=OK
int64: type=integer expect=integer val_one=OK device_cpu=OK
uint64: type=string expect=string val_one=OK device_cpu=OK
float4 sum_of_4=4
float8 sum_of_8=8
float16 sum_of_8=8
float32 sum_of_8=8
float64 sum_of_8=8
float128 sum_of_8=8
int8 sum_of_8=8
uint8 sum_of_8=8
int16 sum_of_8=8
uint16 sum_of_8=8
int32 sum_of_8=8
uint32 sum_of_8=8
int64 sum_of_8=8
uint64 sum_of_8=8
