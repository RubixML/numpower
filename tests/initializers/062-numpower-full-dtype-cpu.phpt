--TEST--
NumPower::full($shape, $value, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* full() must accept every dtype the engine knows about, produce a buffer
   whose elements (a) decode as the requested value, (b) round-trip through
   the dtype-mandated PHP type, and (c) respect string-form fill for the
   wide dtypes (float128 / int64 / uint64). The old NDArray_Full was a
   float32 / CPU only path that silently dropped any other dtype — this
   test pins the new contract. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$expected_type = [
    'float4'   => 'double',  'float8'   => 'double',  'float16'  => 'double',
    'float32'  => 'double',  'float64'  => 'double',  'float128' => 'string',
    'int8'     => 'integer', 'uint8'    => 'integer',
    'int16'    => 'integer', 'uint16'   => 'integer',
    'int32'    => 'integer', 'uint32'   => 'integer',
    'int64'    => 'integer', 'uint64'   => 'string',
];

/* Pick a numeric fill that's representable across every dtype's range. */
foreach ($dtypes as $dt) {
    $a = NumPower::full([3], 2, $dt);
    $v = $a[0];
    $is_two = ($v === 2 || $v === 2.0 || $v === '2' || $v === '2.0');
    echo $dt, ': type=', gettype($v),
         ' expect=', $expected_type[$dt],
         ' val_two=', ($is_two ? 'OK' : 'BAD'),
         ' device_cpu=', ($a->isGPU() ? 'BAD' : 'OK'),
         "\n";
}

/* Sum of `n` twos must equal 2n. uint8 fits 2 * 100 = 200; cap n at 100
   so every dtype's range stays comfortable. */
foreach ($dtypes as $dt) {
    $a = NumPower::full([100], 2, $dt);
    echo $dt, ' sum=', (string)NumPower::sum($a), "\n";
}

/* String fill — the only loss-free path for the widest dtypes. */
echo "fp128 string: ", rtrim((string)NumPower::full([], "1.5", "float128")), "\n";
echo "uint64 string max: ",
     rtrim((string)NumPower::full([], "18446744073709551615", "uint64")), "\n";
echo "int64 string max: ",
     rtrim((string)NumPower::full([], "9223372036854775807", "int64")), "\n";

/* bool fill — supported by the encoder, matches fill(). */
echo "bool true → uint8: ", rtrim((string)NumPower::full([], true, "uint8")), "\n";
echo "bool false → uint8: ", rtrim((string)NumPower::full([], false, "uint8")), "\n";
?>
--EXPECT--
float4: type=double expect=double val_two=OK device_cpu=OK
float8: type=double expect=double val_two=OK device_cpu=OK
float16: type=double expect=double val_two=OK device_cpu=OK
float32: type=double expect=double val_two=OK device_cpu=OK
float64: type=double expect=double val_two=OK device_cpu=OK
float128: type=string expect=string val_two=OK device_cpu=OK
int8: type=integer expect=integer val_two=OK device_cpu=OK
uint8: type=integer expect=integer val_two=OK device_cpu=OK
int16: type=integer expect=integer val_two=OK device_cpu=OK
uint16: type=integer expect=integer val_two=OK device_cpu=OK
int32: type=integer expect=integer val_two=OK device_cpu=OK
uint32: type=integer expect=integer val_two=OK device_cpu=OK
int64: type=integer expect=integer val_two=OK device_cpu=OK
uint64: type=string expect=string val_two=OK device_cpu=OK
float4 sum=200
float8 sum=200
float16 sum=200
float32 sum=200
float64 sum=200
float128 sum=200
int8 sum=200
uint8 sum=200
int16 sum=200
uint16 sum=200
int32 sum=200
uint32 sum=200
int64 sum=200
uint64 sum=200
fp128 string: 1.5
uint64 string max: 18446744073709551615
int64 string max: 9223372036854775807
bool true → uint8: 1
bool false → uint8: 0
