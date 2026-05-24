--TEST--
NumPower::arange($stop, $start, $step, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* arange() must accept every dtype the engine knows about and produce
   the correct closed-form sequence. The old NDArray_Arange was a
   float32 / CPU only path that silently dropped any other dtype — this
   test pins the new contract:
    - leaf-element type matches the dtype-mandated PHP type.
    - shape is [length], length matches the math.
    - sum of arange(0..n-1, dtype) == n*(n-1)/2 for every dtype that
      represents that range exactly. */

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
    /* float4 only encodes {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}; cap n=4. */
    $n = ($dt === 'float4') ? 4 : 8;
    $a = NumPower::arange($n, 0, 1, $dt);
    $shape_ok = $a->shape() === [$n];
    $type_ok  = (gettype($a[0]) === $expected_type[$dt]);
    $first_ok = ((string)$a[0] === '0' || (string)$a[0] === '0.0');
    /* sum = n*(n-1)/2 for arange(0..n-1). */
    $expected_sum = $n * ($n - 1) / 2;
    $sum_ok = ((string)NumPower::sum($a) === (string)$expected_sum);
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' first=', ($first_ok ? 'OK' : 'BAD'),
         ' sum=', ($sum_ok ? 'OK' : 'BAD'), "\n";
}

/* Non-trivial start / step. */
$a = NumPower::arange(20, 10, 2);
echo 'start/step: ', (string)$a, "\n";

/* String forms — full precision for the wide dtypes. */
$a = NumPower::arange('18446744073709551615', '18446744073709551610', '1', 'uint64');
echo 'uint64_str_max: ', (string)$a, "\n";

$a = NumPower::arange('-9223372036854775800', '-9223372036854775808', '1', 'int64');
echo 'int64_str_neg: ', (string)$a, "\n";

$a = NumPower::arange('1.0', '0.0', '0.25', 'float128');
echo 'fp128_str: ', (string)$a, "\n";
?>
--EXPECT--
float4: shape=OK type=OK first=OK sum=OK
float8: shape=OK type=OK first=OK sum=OK
float16: shape=OK type=OK first=OK sum=OK
float32: shape=OK type=OK first=OK sum=OK
float64: shape=OK type=OK first=OK sum=OK
float128: shape=OK type=OK first=OK sum=OK
int8: shape=OK type=OK first=OK sum=OK
uint8: shape=OK type=OK first=OK sum=OK
int16: shape=OK type=OK first=OK sum=OK
uint16: shape=OK type=OK first=OK sum=OK
int32: shape=OK type=OK first=OK sum=OK
uint32: shape=OK type=OK first=OK sum=OK
int64: shape=OK type=OK first=OK sum=OK
uint64: shape=OK type=OK first=OK sum=OK
start/step: [10, 12, 14, 16, 18]
uint64_str_max: [18446744073709551610, 18446744073709551611, 18446744073709551612, 18446744073709551613, 18446744073709551614]
int64_str_neg: [-9223372036854775808, -9223372036854775807, -9223372036854775806, -9223372036854775805, -9223372036854775804, -9223372036854775803, -9223372036854775802, -9223372036854775801]
fp128_str: [0, 0.25, 0.5, 0.75]
