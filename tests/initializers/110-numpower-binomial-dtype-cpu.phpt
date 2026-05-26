--TEST--
NumPower::randomBinomial($shape, $n, $p, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* randomBinomial() must accept every dtype and produce non-negative
   integer counts in `[0, n]`. The old `NDArray_Binomial` was a
   float32 / CPU-only path that silently dropped every other dtype —
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
    /* Pick n that fits the dtype's representable range. Mean of B(n,p)
       is np with stddev sqrt(np(1-p)); we keep np well below any narrow
       dtype's ceiling. */
    if ($dt === 'float4')      { $n_trials = 1; }
    elseif ($dt === 'float8')  { $n_trials = 2; }
    elseif ($dt === 'int8')    { $n_trials = 100; }
    elseif ($dt === 'uint8')   { $n_trials = 200; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $n_trials = 1000; }
    else                       { $n_trials = 50; }

    $a = NumPower::randomBinomial([64], $n_trials, 0.5, $dt);
    $shape_ok  = ($a->shape() === [64]);
    $type_ok   = (gettype($a[0]) === $expected_type[$dt]);
    $device_ok = !$a->isGPU();
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device_cpu=', ($device_ok ? 'OK' : 'BAD'), "\n";
}

/* Distribution moments. For B(n, p), mean = np and variance = np(1-p).
   With N = 16384 samples and modest n, the standard error of the
   sample mean is sqrt(np(1-p)/N) ≈ 0.04 for n=100, p=0.5, so a 0.5
   tolerance is overwhelming probability. */
function php_mean_var($a) {
    $arr = $a->toArray();
    $sum = 0.0; $cnt = 0;
    foreach ($arr as $v) { $sum += (float)$v; $cnt++; }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) { $d = ((float)$v) - $mean; $sq += $d * $d; }
    return [$mean, $sq / $cnt];
}

$N = 16384;

/* B(10, 0.5) — classic coin-flip. */
$a = NumPower::randomBinomial([$N], 10, 0.5, 'float32');
[$mean, $var] = php_mean_var($a);
$mean_ok = abs($mean - 5.0) < 0.2;
$var_ok  = abs($var  - 2.5) < 0.3;
echo 'dist_B10_0.5: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* B(100, 0.3) — typical sampling task. */
$a = NumPower::randomBinomial([$N], 100, 0.3, 'int32');
[$mean, $var] = php_mean_var($a);
$mean_ok = abs($mean - 30.0) < 1.0;
$var_ok  = abs($var  - 21.0) < 2.0;
echo 'dist_B100_0.3: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* B(1000, 0.5) — exercises the O(n) loop at moderate scale. With
   N=4096, SE(var) ≈ sqrt(2)·250/sqrt(4095) ≈ 5.5, so use a ~5σ
   tolerance (~30) for the variance check. */
$a = NumPower::randomBinomial([4096], 1000, 0.5, 'float64');
[$mean, $var] = php_mean_var($a);
$mean_ok = abs($mean - 500.0) < 3.0;
$var_ok  = abs($var  - 250.0) < 30.0;
echo 'dist_B1000_0.5: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* Counts must be non-negative integers in `[0, n]`. */
$a = NumPower::randomBinomial([2048], 20, 0.5, 'float64');
$range_ok = true;
foreach ($a->toArray() as $v) {
    $f = (float)$v;
    if ($f < 0.0 || $f > 20.0 || $f !== floor($f)) { $range_ok = false; break; }
}
echo 'count_range_B20: ', ($range_ok ? 'OK' : 'BAD'), "\n";

/* p = 1.0 — every sample must equal n exactly (the legacy bug from
   `rand() / RAND_MAX` returning 1.0 would occasionally undercount
   here). */
$a = NumPower::randomBinomial([256], 50, 1.0, 'int32');
$all_n = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 50) { $all_n = false; break; }
}
echo 'p_eq_one_all_n: ', ($all_n ? 'OK' : 'BAD'), "\n";

/* p = 0.0 — every sample must equal 0 exactly. */
$a = NumPower::randomBinomial([256], 50, 0.0, 'int32');
$all_zero = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'p_eq_zero_all_zero: ', ($all_zero ? 'OK' : 'BAD'), "\n";

/* n = 0 — every sample must equal 0 across dtypes. */
$all_zero = true;
foreach (['float32', 'int32', 'uint64', 'float128'] as $dt) {
    $a = NumPower::randomBinomial([16], 0, 0.5, $dt);
    foreach ($a->toArray() as $v) {
        if ((float)$v !== 0.0) { $all_zero = false; break 2; }
    }
}
echo 'n_eq_zero_degenerate: ', ($all_zero ? 'OK' : 'BAD'), "\n";
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
dist_B10_0.5: mean_ok=OK var_ok=OK
dist_B100_0.3: mean_ok=OK var_ok=OK
dist_B1000_0.5: mean_ok=OK var_ok=OK
count_range_B20: OK
p_eq_one_all_n: OK
p_eq_zero_all_zero: OK
n_eq_zero_degenerate: OK
