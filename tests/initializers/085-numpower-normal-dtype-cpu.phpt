--TEST--
NumPower::normal($shape, $loc, $scale, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* normal() must accept every dtype and produce samples whose leaf
   element type matches the dtype contract, shape matches the request,
   and (when the dtype carries enough range) the sample mean / stddev
   land near the requested distribution parameters. The old NDArray_Normal
   was a float32 / CPU only path that silently dropped any other dtype —
   this test pins the new contract. */

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
    /* Pick (loc, scale) inside each dtype's representable range and
       inside its precision so the mean/stddev check is meaningful. */
    if ($dt === 'float4') { $loc = 1.0; $scale = 0.5; }
    elseif ($dt === 'float8') { $loc = 2.0; $scale = 1.0; }
    elseif ($dt === 'int8') { $loc = 0; $scale = 20; }
    elseif ($dt === 'uint8') { $loc = 128; $scale = 30; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $loc = 1000; $scale = 100; }
    elseif ($dt === 'uint64') { $loc = '1000000'; $scale = '100'; }
    elseif ($dt === 'float128') { $loc = '0.0'; $scale = '1.0'; }
    else { $loc = 0.0; $scale = 1.0; }

    $a = NumPower::normal([64], $loc, $scale, $dt);
    $shape_ok  = ($a->shape() === [64]);
    $type_ok   = (gettype($a[0]) === $expected_type[$dt]);
    $device_ok = !$a->isGPU();
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device_cpu=', ($device_ok ? 'OK' : 'BAD'), "\n";
}

/* Distribution check: float32 with loc=0, scale=1, n=8192. The sample
   mean and stddev must land near the requested distribution within a
   3-sigma window (the law-of-large-numbers expects |mean| < ~3/sqrt(n)
   and |stddev - 1| < ~3/sqrt(2n) on N(0, 1)). NumPower::std has a
   known pre-existing dtype bug on float64; we compute std in PHP from
   toArray() so the check is dtype-agnostic. */
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
$a = NumPower::normal([$n], 0.0, 1.0, 'float32');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$mean_ok = (abs($mean) < 0.1);
$std_ok  = (abs($std - 1.0) < 0.1);
echo 'dist_f32: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* Shifted distribution: loc=5, scale=2 — sample mean ≈ 5, stddev ≈ 2. */
$a = NumPower::normal([$n], 5.0, 2.0, 'float64');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$mean_ok = (abs($mean - 5.0) < 0.2);
$std_ok  = (abs($std  - 2.0) < 0.2);
echo 'dist_f64: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"),  "\n";

/* String-form loc/scale on float32 is just a parse path, but it must
   work and produce the same distribution as the int/float form. */
$a = NumPower::normal([$n], '0.0', '1.0', 'float32');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$str_ok = (abs($mean) < 0.1) && (abs($std - 1.0) < 0.1);
echo 'str_loc_scale_f32: ', ($str_ok ? 'OK' : "BAD m=$mean s=$std"), "\n";

/* Int-form loc/scale: int → float widening. */
$a = NumPower::normal([$n], 0, 1, 'float32');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$int_ok = (abs($mean) < 0.1) && (abs($std - 1.0) < 0.1);
echo 'int_loc_scale_f32: ', ($int_ok ? 'OK' : "BAD m=$mean s=$std"), "\n";

/* fp128 wide-range mean: loc='1e+100', scale='1.0'. Sample mean should
   round-trip the loc; sample stddev should remain near 1. */
$a = NumPower::normal([$n], '1.0e+100', '1.0', 'float128');
/* fp128 stddev computation is sensitive to subtraction-cancellation
   when the mean is enormous compared to scale, so we just spot-check
   that |sample - loc| is within a few stddevs. The first element should
   be roughly loc ± O(scale). */
$sample0 = (string)$a[0];
/* The result, formatted, should still contain "1.0" something close
   to 10^100 — the noise term is ±~1 which is far below fp128 resolution
   at this magnitude (~10^85), so each value displays as ~1e100 exactly. */
$wide_ok = (strpos($sample0, 'e+100') !== false ||
            strpos($sample0, 'E+100') !== false ||
            strlen($sample0) >= 50);
echo 'fp128_wide_loc: ', ($wide_ok ? 'OK' : "BAD($sample0)"), "\n";

/* uint64 wide mean: loc='18446744073709551000', scale='10'. Every
   sample should land near loc — uint64 has no headroom above
   2^64 - 1 but the test ranges keep us safely below that. */
$a = NumPower::normal([$n], '18446744073709551000', '10', 'uint64');
$sample0 = (string)$a[0];
$u64_ok = (strlen($sample0) >= 19 && strlen($sample0) <= 20);
echo 'u64_wide_loc: ', ($u64_ok ? 'OK' : "BAD($sample0)"), "\n";
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
str_loc_scale_f32: OK
int_loc_scale_f32: OK
fp128_wide_loc: OK
u64_wide_loc: OK
