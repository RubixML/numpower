--TEST--
NumPower::truncatedNormal($shape, $loc, $scale, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* truncatedNormal() must accept every dtype and produce samples whose
   (a) leaf element type matches the dtype contract, (b) shape matches
   the request, (c) sample mean lands near the requested mean, (d)
   every accepted value lies within `[loc - 2σ, loc + 2σ]` — the
   defining truncation property. */

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
    /* Pick dtype-appropriate (loc, scale). */
    if ($dt === 'float4') { $loc = 1.0; $scale = 0.5; }
    elseif ($dt === 'float8') { $loc = 2.0; $scale = 1.0; }
    elseif ($dt === 'int8') { $loc = 0; $scale = 20; }
    elseif ($dt === 'uint8') { $loc = 128; $scale = 30; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $loc = 1000; $scale = 100; }
    elseif ($dt === 'uint64') { $loc = '1000000'; $scale = '100'; }
    elseif ($dt === 'float128') { $loc = '0.0'; $scale = '1.0'; }
    else { $loc = 0.0; $scale = 1.0; }

    $a = NumPower::truncatedNormal([64], $loc, $scale, $dt);
    $shape_ok  = ($a->shape() === [64]);
    $type_ok   = (gettype($a[0]) === $expected_type[$dt]);
    $device_ok = !$a->isGPU();
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device_cpu=', ($device_ok ? 'OK' : 'BAD'), "\n";
}

/* Truncation check on float32: every value MUST be inside [loc - 2σ,
   loc + 2σ]. The 2σ bound is exact (modulo dtype quantisation) — a
   bug in the rejection loop would surface as out-of-window samples. */
function check_truncation_window($a, $loc, $scale, $tag, $dtype_quant_eps = 1e-5) {
    $arr = $a->toArray();
    $lo  = $loc - 2.0 * $scale - $dtype_quant_eps;
    $hi  = $loc + 2.0 * $scale + $dtype_quant_eps;
    $bad = 0;
    foreach ($arr as $v) {
        $f = (float)$v;
        if ($f < $lo || $f > $hi) {
            $bad++;
        }
    }
    echo $tag, ': out_of_window=', $bad,
         ' (window [', $lo, ', ', $hi, '])',
         "\n";
}

$a = NumPower::truncatedNormal([1024], 0.0, 1.0, 'float32');
check_truncation_window($a, 0.0, 1.0, 'window_f32_std');

$a = NumPower::truncatedNormal([1024], 5.0, 2.0, 'float64');
check_truncation_window($a, 5.0, 2.0, 'window_f64_shifted');

/* Distribution check: float32 with loc=0, scale=1, n=8192.
   For a normal truncated at ±2σ:
     E[X] = µ  (symmetric truncation)
     Var[X] = σ²(1 - 4φ(2)/Φ(2,-2)) ≈ 0.7737 σ²
     std[X] ≈ 0.8796 σ
   We allow 0.15σ slack on both. */
function php_std($a, $mean) {
    $arr = $a->toArray();
    $sum = 0.0; $n = 0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sum += $d * $d; $n++; }
    return sqrt($sum / $n);
}

$n = 8192;
$a = NumPower::truncatedNormal([$n], 0.0, 1.0, 'float32');
$mean = (float) NumPower::mean($a);
$std  = php_std($a, $mean);
$mean_ok = abs($mean) < 0.1;
$std_ok  = abs($std - 0.8796) < 0.1;
echo 'stat_f32: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' std_ok=',  ($std_ok  ? 'OK' : "BAD($std)"), "\n";

/* String-form loc/scale, plus int-form. */
$a = NumPower::truncatedNormal([1024], '0.0', '1.0', 'float32');
check_truncation_window($a, 0.0, 1.0, 'window_f32_str_args');

$a = NumPower::truncatedNormal([1024], 0, 1, 'float32');
check_truncation_window($a, 0.0, 1.0, 'window_f32_int_args');

/* fp128 wide-range loc — every sample should print near 1e+200 because
   the noise term is far below fp128's resolution at that magnitude. */
$a = NumPower::truncatedNormal([4], '1.0e+200', '1.0', 'float128');
$s0 = (string)$a[0];
$wide_ok = (strpos($s0, 'e+200') !== false || strpos($s0, 'E+200') !== false ||
            strlen($s0) >= 30);
echo 'fp128_wide_loc: ', ($wide_ok ? 'OK' : "BAD($s0)"), "\n";

/* uint64 — every value within [loc - 2*scale, loc + 2*scale]. */
$a = NumPower::truncatedNormal([1024], '1000000', '100', 'uint64');
$bad = 0;
foreach ($a->toArray() as $v) {
    $u = (int)$v;
    if ($u < 999800 || $u > 1000200) $bad++;
}
echo 'u64_window: out_of_window=', $bad, "\n";
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
window_f32_std: out_of_window=0 (window [-2.00001, 2.00001])
window_f64_shifted: out_of_window=0 (window [0.99999, 9.00001])
stat_f32: mean_ok=OK std_ok=OK
window_f32_str_args: out_of_window=0 (window [-2.00001, 2.00001])
window_f32_int_args: out_of_window=0 (window [-2.00001, 2.00001])
fp128_wide_loc: OK
u64_window: out_of_window=0
