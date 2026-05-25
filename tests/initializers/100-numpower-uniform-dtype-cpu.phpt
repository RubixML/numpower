--TEST--
NumPower::uniform($shape, $low, $high, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* uniform() must accept every dtype and produce samples whose leaf
   element type matches the dtype contract, whose shape matches the
   request, and (for dtypes wide enough to represent the requested
   range) whose values land in `[low, high)`. The old NDArray_Uniform
   was a float32 / CPU-only path that silently dropped every other
   dtype — this test pins the new contract. */

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

mt_srand(42);
srand(42);

foreach ($dtypes as $dt) {
    /* Pick (low, high) inside each dtype's representable range so the
       range / type check is meaningful. */
    if ($dt === 'float4') { $low = 0.0; $high = 1.0; }
    elseif ($dt === 'float8') { $low = 0.0; $high = 2.0; }
    elseif ($dt === 'int8') { $low = 0; $high = 100; }
    elseif ($dt === 'uint8') { $low = 0; $high = 255; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $low = 0; $high = 1000; }
    elseif ($dt === 'uint64') { $low = '1000000'; $high = '2000000'; }
    elseif ($dt === 'float128') { $low = '0.0'; $high = '1.0'; }
    else { $low = 0.0; $high = 1.0; }

    $a = NumPower::uniform([64], $low, $high, $dt);
    $shape_ok  = ($a->shape() === [64]);
    $type_ok   = (gettype($a[0]) === $expected_type[$dt]);
    $device_ok = !$a->isGPU();
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device_cpu=', ($device_ok ? 'OK' : 'BAD'), "\n";
}

/* Distribution check on float32: U([0, 1)) over n=16384 samples should
   give mean near 0.5 and standard error of 1/sqrt(12 * 16384) ≈ 0.0023.
   A 0.05 tolerance passes with overwhelming probability. The actual
   range must also be closed at 0 and strictly open at 1. */
$n = 16384;
$a = NumPower::uniform([$n], 0.0, 1.0, 'float32');
$arr = $a->toArray();
$min = INF; $max = -INF; $sum = 0.0;
foreach ($arr as $v) {
    $f = (float)$v;
    if ($f < $min) $min = $f;
    if ($f > $max) $max = $f;
    $sum += $f;
}
$mean    = $sum / $n;
$mean_ok = (abs($mean - 0.5) < 0.05);
$range_ok = ($min >= 0.0 && $max < 1.0);
echo 'dist_f32: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' range_ok=',  ($range_ok ? 'OK' : "BAD($min..$max)"),  "\n";

/* Shifted distribution on float64: U([-10, 10)). Mean ≈ 0, range
   strict. */
$a = NumPower::uniform([$n], -10.0, 10.0, 'float64');
$arr = $a->toArray();
$min = INF; $max = -INF; $sum = 0.0;
foreach ($arr as $v) {
    $f = (float)$v;
    if ($f < $min) $min = $f;
    if ($f > $max) $max = $f;
    $sum += $f;
}
$mean     = $sum / $n;
$mean_ok  = (abs($mean) < 0.5);
$range_ok = ($min >= -10.0 && $max < 10.0);
echo 'dist_f64_shifted: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' range_ok=',  ($range_ok ? 'OK' : "BAD($min..$max)"),  "\n";

/* String-form low/high on float32: same distribution as int/float forms. */
$a = NumPower::uniform([$n], '0.0', '1.0', 'float32');
$arr = $a->toArray();
$sum = 0.0;
foreach ($arr as $v) $sum += (float)$v;
$mean = $sum / $n;
echo 'str_low_high_f32: ', (abs($mean - 0.5) < 0.05 ? 'OK' : "BAD($mean)"), "\n";

/* Int-form low/high on float32: int widens to double cleanly. */
$a = NumPower::uniform([$n], 0, 1, 'float32');
$arr = $a->toArray();
$sum = 0.0;
foreach ($arr as $v) $sum += (float)$v;
$mean = $sum / $n;
echo 'int_low_high_f32: ', (abs($mean - 0.5) < 0.05 ? 'OK' : "BAD($mean)"), "\n";

/* fp128 wide-range bounds: low='1e+100', high='2e+100'. Each sample
   should fall in that range; the leading digit (1) should also vary
   above and below 1.5e+100. Spot-check the first sample's order. */
$a = NumPower::uniform([$n], '1.0e+100', '2.0e+100', 'float128');
$sample0 = (string)$a[0];
$fp128_wide_ok = (strpos($sample0, 'e+100') !== false ||
                  strpos($sample0, 'E+100') !== false ||
                  strlen($sample0) >= 50);
echo 'fp128_wide: ', ($fp128_wide_ok ? 'OK' : "BAD($sample0)"), "\n";

/* uint64 wide-range bounds: high='18446744073709550000', low='10'.
   Every sample's string must have ≥ 18 digits to land in that range. */
$a = NumPower::uniform([$n], '10', '18446744073709550000', 'uint64');
$sample0 = (string)$a[0];
$u64_wide_ok = (strlen($sample0) >= 18 && strlen($sample0) <= 20);
echo 'u64_wide: ', ($u64_wide_ok ? 'OK' : "BAD($sample0)"), "\n";

/* int8 range check — values must land in [0, 100). */
$a = NumPower::uniform([512], 0, 100, 'int8');
$arr  = $a->toArray();
$ok   = true;
foreach ($arr as $v) {
    if (!is_int($v) || $v < 0 || $v >= 100) { $ok = false; break; }
}
echo 'int8_range_check: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float4: shape=OK type=OK device_cpu=OK
float8: shape=OK type=OK device_cpu=OK
float16: shape=OK type=OK device_cpu=OK
float32: shape=OK type=OK device_cpu=OK
float64: shape=OK type=OK device_cpu=OK
float128: shape=OK type=OK device_cpu=OK
int8: shape=OK type=OK device_cpu=OK
uint8: shape=OK type=OK device_cpu=OK
int16: shape=OK type=OK device_cpu=OK
uint16: shape=OK type=OK device_cpu=OK
int32: shape=OK type=OK device_cpu=OK
uint32: shape=OK type=OK device_cpu=OK
int64: shape=OK type=OK device_cpu=OK
uint64: shape=OK type=OK device_cpu=OK
dist_f32: mean_ok=OK range_ok=OK
dist_f64_shifted: mean_ok=OK range_ok=OK
str_low_high_f32: OK
int_low_high_f32: OK
fp128_wide: OK
u64_wide: OK
int8_range_check: OK
