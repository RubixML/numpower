--TEST--
NumPower::standardNormal($shape, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* standardNormal() must accept every dtype and produce samples whose
   leaf element type matches the dtype contract, whose shape matches
   the request, and (for dtypes wide enough to represent N(0, 1)) whose
   sample mean / stddev land near 0 and 1. The legacy entry point was
   float32 / CPU only and silently dropped any other dtype — this test
   pins the new contract. */

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
    $a = NumPower::standardNormal([64], $dt);
    $shape_ok  = ($a->shape() === [64]);
    $type_ok   = (gettype($a[0]) === $expected_type[$dt]);
    $device_ok = !$a->isGPU();
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device_cpu=', ($device_ok ? 'OK' : 'BAD'), "\n";
}

/* Distribution check: float32 standard-normal, n=8192. Sample mean /
   stddev must land near N(0, 1) within a ~3-sigma window. Compute std
   in PHP from toArray() since NumPower::std has a pre-existing dtype
   bug on float64. */
function php_std($a, $mean) {
    $arr = $a->toArray();
    $sum = 0.0;
    $n   = 0;
    foreach ($arr as $v) {
        $d = ((float)$v) - $mean;
        $sum += $d * $d;
        $n++;
    }
    return sqrt($sum / $n);
}

$n = 8192;
$a = NumPower::standardNormal([$n], 'float32');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$mean_ok = (abs($mean) < 0.1);
$std_ok  = (abs($std - 1.0) < 0.1);
echo 'dist_f32: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* Same distribution on float64. */
$a = NumPower::standardNormal([$n], 'float64');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$mean_ok = (abs($mean) < 0.1);
$std_ok  = (abs($std - 1.0) < 0.1);
echo 'dist_f64: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* float128 standard-normal: the (loc=0, scale=1) defaults are encoded
   in fp128 so the on-device DD-affine path stays bit-exact. We only
   spot-check that the result decodes via the string-typed accessor
   and the magnitudes are plausible (almost all samples should land in
   [-5, +5]). */
$a = NumPower::standardNormal([16], 'float128');
$ok = true;
foreach ($a->toArray() as $v) {
    $f = (float) $v;
    if (!is_finite($f) || abs($f) > 50.0) { $ok = false; break; }
}
echo 'fp128_range: ', ($ok ? 'OK' : 'BAD'), "\n";

/* 68-95-99.7 rule on float32: ~68% of samples land within 1σ of the
   mean. With n=8192 the empirical fraction should be in [0.55, 0.80]
   with overwhelming probability — comfortably loose tolerance. */
$a = NumPower::standardNormal([$n], 'float32');
$mean = 0.0;
$cnt = 0;
foreach ($a->toArray() as $v) {
    if (abs((float)$v) < 1.0) $cnt++;
}
$frac = $cnt / $n;
$rule_ok = $frac > 0.55 && $frac < 0.80;
echo '68_pct_rule_f32: ', ($rule_ok ? 'OK' : "BAD($frac)"), "\n";
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
dist_f32: mean_ok=OK std_ok=OK
dist_f64: mean_ok=OK std_ok=OK
fp128_range: OK
68_pct_rule_f32: OK
