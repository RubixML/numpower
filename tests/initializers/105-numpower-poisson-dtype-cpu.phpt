--TEST--
NumPower::poisson($shape, $lam, $dtype) covers every supported dtype on CPU
--FILE--
<?php
/* poisson() must accept every dtype and produce non-negative integer
   counts whose leaf element type matches the dtype contract. The old
   `NDArray_Poisson` was a float32 / CPU-only path that silently dropped
   every other dtype — this test pins the new contract. */

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
    /* Pick a lam that fits the dtype's representable range. Most
       Poisson samples land in `[lam - 5*sqrt(lam), lam + 5*sqrt(lam)]`
       so we keep lam well below any narrow dtype's ceiling. */
    if ($dt === 'float4')      { $lam = 1.0; }
    elseif ($dt === 'float8')  { $lam = 2.0; }
    elseif ($dt === 'int8')    { $lam = 30.0; }
    elseif ($dt === 'uint8')   { $lam = 100.0; }
    elseif ($dt === 'int16' || $dt === 'uint16') { $lam = 1000.0; }
    elseif ($dt === 'uint64')  { $lam = '1000000'; }
    elseif ($dt === 'float128'){ $lam = '50.0'; }
    else                       { $lam = 5.0; }

    $a = NumPower::poisson([64], $lam, $dt);
    $shape_ok  = ($a->shape() === [64]);
    $type_ok   = (gettype($a[0]) === $expected_type[$dt]);
    $device_ok = !$a->isGPU();
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' device_cpu=', ($device_ok ? 'OK' : 'BAD'), "\n";
}

/* Distribution moments. For Poisson(λ), mean = variance = λ. With
   N = 16384 samples and λ = 10, the standard error of the mean is
   sqrt(10/16384) ≈ 0.025, so a 0.5 tolerance is overwhelming
   probability. */
function php_mean_var($a) {
    $arr = $a->toArray();
    $sum = 0.0;
    $cnt = 0;
    foreach ($arr as $v) {
        $sum += (float)$v;
        $cnt++;
    }
    $mean = $sum / $cnt;
    $sq = 0.0;
    foreach ($arr as $v) {
        $d   = ((float)$v) - $mean;
        $sq += $d * $d;
    }
    return [$mean, $sq / $cnt];
}

$n = 16384;

/* Knuth branch (lam < 30). */
$a = NumPower::poisson([$n], 5.0, 'float32');
[$mean, $var] = php_mean_var($a);
$mean_ok = abs($mean - 5.0) < 0.5;
$var_ok  = abs($var  - 5.0) < 1.0;
echo 'knuth_lam5: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* PTRS branch (lam >= 30). */
$a = NumPower::poisson([$n], 100.0, 'int32');
[$mean, $var] = php_mean_var($a);
$mean_ok = abs($mean - 100.0) < 2.0;
$var_ok  = abs($var  - 100.0) < 5.0;
echo 'ptrs_lam100: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* PTRS branch with large lam — the legacy `expf(-lam)` underflow bug
   would have hung the process forever here. */
$a = NumPower::poisson([$n], 200.0, 'int32');
[$mean, $var] = php_mean_var($a);
$mean_ok = abs($mean - 200.0) < 3.0;
$var_ok  = abs($var  - 200.0) < 10.0;
echo 'ptrs_lam200: mean_ok=', ($mean_ok ? 'OK' : "BAD($mean)"),
     ' var_ok=', ($var_ok ? 'OK' : "BAD($var)"), "\n";

/* Counts must be non-negative integers — for any dtype the leaf
   element decodes to a value with `v == floor(v)` and `v >= 0`. */
$a = NumPower::poisson([1024], 5.0, 'float64');
$nonneg_int = true;
foreach ($a->toArray() as $v) {
    $f = (float)$v;
    if ($f < 0.0 || $f !== floor($f)) { $nonneg_int = false; break; }
}
echo 'nonneg_integer_counts: ', ($nonneg_int ? 'OK' : 'BAD'), "\n";

/* lam = 0 is degenerate — every sample must be 0 across every dtype. */
$all_zero = true;
foreach (['float32', 'int32', 'uint64', 'float128'] as $dt) {
    $a = NumPower::poisson([16], 0, $dt);
    foreach ($a->toArray() as $v) {
        if ((float)$v !== 0.0) { $all_zero = false; break 2; }
    }
}
echo 'lam_zero_degenerate: ', ($all_zero ? 'OK' : 'BAD'), "\n";

/* String / int / float forms of lam all produce the same distribution. */
foreach ([5, 5.0, '5.0', '5'] as $lam_form) {
    $a = NumPower::poisson([4096], $lam_form, 'float32');
    [$m, $_] = php_mean_var($a);
    if (abs($m - 5.0) >= 0.5) {
        echo "lam_form mismatch ($lam_form): mean=$m\n";
    }
}
echo 'lam_form_parity: OK', "\n";
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
knuth_lam5: mean_ok=OK var_ok=OK
ptrs_lam100: mean_ok=OK var_ok=OK
ptrs_lam200: mean_ok=OK var_ok=OK
nonneg_integer_counts: OK
lam_zero_degenerate: OK
lam_form_parity: OK
