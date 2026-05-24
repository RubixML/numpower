--TEST--
NumPower::identity($size, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* identity() must produce a correct N×N identity matrix for every dtype,
   with the dtype-appropriate "1" on the diagonal and "0" elsewhere. The
   old NDArray_Identity hardcoded float32 storage and broadcast a float
   1.0 into the buffer — this test pins the new contract:
    - shape is always [N, N] (a 2-D matrix, not a 1-D collapsing for N=0).
    - leaf-element type matches the dtype-mandated PHP type.
    - sum equals N (trace == N for any identity).
    - off-diagonal elements are exactly zero. */

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

foreach ($dtypes as $dt) {
    $a = NumPower::identity(4, $dt);
    $shape_ok = $a->shape() === [4, 4];
    $diag = $a[0][0];
    $off  = $a[0][1];
    $is_one = ($diag === 1 || $diag === 1.0 || $diag === '1' || $diag === '1.0');
    $is_zero = ($off === 0 || $off === 0.0 || $off === '0' || $off === '0.0');
    $type_ok = (gettype($diag) === $expected_type[$dt]);
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' diag_one=', ($is_one ? 'OK' : 'BAD'),
         ' off_zero=', ($is_zero ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device=', ($a->isGPU() ? 'BAD' : 'OK'),
         "\n";
}

/* Trace test: sum of an N×N identity is N for every dtype. */
foreach ($dtypes as $dt) {
    $a = NumPower::identity(8, $dt);
    echo $dt, ' trace=', (string)NumPower::sum($a), "\n";
}
?>
--EXPECT--
float4: shape=OK diag_one=OK off_zero=OK type=OK device=OK
float8: shape=OK diag_one=OK off_zero=OK type=OK device=OK
float16: shape=OK diag_one=OK off_zero=OK type=OK device=OK
float32: shape=OK diag_one=OK off_zero=OK type=OK device=OK
float64: shape=OK diag_one=OK off_zero=OK type=OK device=OK
float128: shape=OK diag_one=OK off_zero=OK type=OK device=OK
int8: shape=OK diag_one=OK off_zero=OK type=OK device=OK
uint8: shape=OK diag_one=OK off_zero=OK type=OK device=OK
int16: shape=OK diag_one=OK off_zero=OK type=OK device=OK
uint16: shape=OK diag_one=OK off_zero=OK type=OK device=OK
int32: shape=OK diag_one=OK off_zero=OK type=OK device=OK
uint32: shape=OK diag_one=OK off_zero=OK type=OK device=OK
int64: shape=OK diag_one=OK off_zero=OK type=OK device=OK
uint64: shape=OK diag_one=OK off_zero=OK type=OK device=OK
float4 trace=8
float8 trace=8
float16 trace=8
float32 trace=8
float64 trace=8
float128 trace=8
int8 trace=8
uint8 trace=8
int16 trace=8
uint16 trace=8
int32 trace=8
uint32 trace=8
int64 trace=8
uint64 trace=8
