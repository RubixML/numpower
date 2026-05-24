--TEST--
NumPower::zeros($shape, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* zeros() must accept every dtype the engine knows about and produce a
   buffer whose elements (a) decode as zero and (b) round-trip through the
   dtype-mandated PHP type. The element-type check guards against accidental
   float32 dispatch when a different dtype was requested. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

/* Expected PHP type returned by `$a[i]` per dtype. Strings only for
   float128 / uint64 (the dtypes that can't fit in PHP's native double / int);
   every other integer dtype returns int, every other float dtype returns
   float (a.k.a. PHP's "double"). Mirrors the contract checked by
   tests/types/025-foreach-dtype-preservation.phpt. */
$expected_type = [
    'float4'   => 'double',
    'float8'   => 'double',
    'float16'  => 'double',
    'float32'  => 'double',
    'float64'  => 'double',
    'float128' => 'string',
    'int8'     => 'integer',
    'uint8'    => 'integer',
    'int16'    => 'integer',
    'uint16'   => 'integer',
    'int32'    => 'integer',
    'uint32'   => 'integer',
    'int64'    => 'integer',
    'uint64'   => 'string',
];

foreach ($dtypes as $dt) {
    $a = NumPower::zeros([3], $dt);
    $v = $a[0];
    /* Compare against both PHP-side zero representations the dtype may pick. */
    $is_zero = ($v === 0 || $v === 0.0 || $v === '0' || $v === '0.0');
    echo $dt, ': type=', gettype($v),
         ' expect=', $expected_type[$dt],
         ' val_zero=', ($is_zero ? 'OK' : 'BAD'),
         ' device_cpu=', ($a->isGPU() ? 'BAD' : 'OK'),
         "\n";
}

/* Sum-of-zeros must be exactly 0 for every dtype. */
foreach ($dtypes as $dt) {
    $a = NumPower::zeros([8], $dt);
    echo $dt, ' sum=', (string)NumPower::sum($a), "\n";
}
?>
--EXPECT--
float4: type=double expect=double val_zero=OK device_cpu=OK
float8: type=double expect=double val_zero=OK device_cpu=OK
float16: type=double expect=double val_zero=OK device_cpu=OK
float32: type=double expect=double val_zero=OK device_cpu=OK
float64: type=double expect=double val_zero=OK device_cpu=OK
float128: type=string expect=string val_zero=OK device_cpu=OK
int8: type=integer expect=integer val_zero=OK device_cpu=OK
uint8: type=integer expect=integer val_zero=OK device_cpu=OK
int16: type=integer expect=integer val_zero=OK device_cpu=OK
uint16: type=integer expect=integer val_zero=OK device_cpu=OK
int32: type=integer expect=integer val_zero=OK device_cpu=OK
uint32: type=integer expect=integer val_zero=OK device_cpu=OK
int64: type=integer expect=integer val_zero=OK device_cpu=OK
uint64: type=string expect=string val_zero=OK device_cpu=OK
float4 sum=0
float8 sum=0
float16 sum=0
float32 sum=0
float64 sum=0
float128 sum=0
int8 sum=0
uint8 sum=0
int16 sum=0
uint16 sum=0
int32 sum=0
uint32 sum=0
int64 sum=0
uint64 sum=0
