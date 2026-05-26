--TEST--
NumPower::poisson() handles boundary shapes and λ values across every dtype
--FILE--
<?php
/* Boundary checks: 0-D shape, [0] shape, [1] shape, multi-dim,
   NDArray-as-shape, the default contract, and the algorithm switch
   points (`lam = 0`, `lam < 30` Knuth, `lam >= 30` PTRS, `lam == 30`
   exactly at the boundary). */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

/* 0-D shape (size = 1, ndim = 0). The factory must return an NDArray
   (not collapse to a primitive zval). */
foreach ($dtypes as $dt) {
    $a = NumPower::poisson([], 5.0, $dt);
    $is_ndarray = ($a instanceof NDArray);
    $shape_ok   = $a->shape() === [];
    echo $dt, ' 0d: ndarray=', ($is_ndarray ? 'OK' : 'BAD'),
         ' shape=', ($shape_ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [0] — zero-element 1-D. */
foreach ($dtypes as $dt) {
    $a = NumPower::poisson([0], 5.0, $dt);
    $ok = $a->shape() === [0] && $a->size() === 0;
    echo $dt, ' zero: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [1]. */
foreach ($dtypes as $dt) {
    $a = NumPower::poisson([1], 5.0, $dt);
    $ok = $a->shape() === [1] && $a->size() === 1;
    echo $dt, ' one: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Multi-dim 3-D shape. */
$a = NumPower::poisson([2, 3, 4], 5.0, 'float32');
echo 'multidim: ', ($a->shape() === [2, 3, 4] && $a->size() === 24 ? 'OK' : 'BAD'), "\n";

/* Shape passed as an NDArray. */
$shape_nd = new NDArray([4, 4]);
$a = NumPower::poisson($shape_nd, 5.0, 'float32');
echo 'shape_nd: ', ($a->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";

/* Defaults — only the shape: lam=1, dtype=float32, device=CPU. */
$a = NumPower::poisson([8]);
echo 'defaults: shape=', implode(',', $a->shape()),
     ' dtype=', (gettype($a[0]) === 'double' ? 'OK' : 'BAD'),
     ' device=', ($a->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Scalar shape — ZVAL_TO_NDARRAY converts to a 1-D vector. */
$a = NumPower::poisson(5, 3.0, 'float32');
$ok = $a->shape() === [5] && $a->size() === 5;
echo 'scalar_shape: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Algorithm switch boundary: lam = 30.0 exactly is the PTRS threshold;
   lam = 29.999 stays in Knuth. Both must produce correct distributions. */
$n = 4096;
function php_mean($a) {
    $s = 0.0; $c = 0;
    foreach ($a->toArray() as $v) { $s += (float)$v; $c++; }
    return $s / $c;
}

$a = NumPower::poisson([$n], 29.999, 'float32');
$mean = php_mean($a);
echo 'knuth_edge_lam29.999: ', (abs($mean - 29.999) < 1.0 ? 'OK' : "BAD($mean)"), "\n";

$a = NumPower::poisson([$n], 30.0, 'float32');
$mean = php_mean($a);
echo 'ptrs_edge_lam30: ', (abs($mean - 30.0) < 1.0 ? 'OK' : "BAD($mean)"), "\n";

/* Large lam — would have hung the legacy `expf(-lam)` implementation. */
$a = NumPower::poisson([$n], 500.0, 'int32');
$mean = php_mean($a);
echo 'large_lam_no_hang: ', (abs($mean - 500.0) < 5.0 ? 'OK' : "BAD($mean)"), "\n";

/* Very small fractional lam — Knuth's first uniform draw must be
   compared against `exp(-0.001)` ≈ 0.999, so the rejection rate is
   high but finite. */
$a = NumPower::poisson([$n], 0.001, 'float32');
$mean = php_mean($a);
echo 'tiny_lam_0.001: ', (abs($mean - 0.001) < 0.05 ? 'OK' : "BAD($mean)"), "\n";

/* lam passed as int. */
$a = NumPower::poisson([$n], 7, 'float32');
$mean = php_mean($a);
echo 'int_lam_7: ', (abs($mean - 7.0) < 0.3 ? 'OK' : "BAD($mean)"), "\n";

/* lam passed as numeric string. */
$a = NumPower::poisson([$n], '12.5', 'float64');
$mean = php_mean($a);
echo 'string_lam_12.5: ', (abs($mean - 12.5) < 0.5 ? 'OK' : "BAD($mean)"), "\n";

/* Narrow dtype + large lam: uint8 with lam=200 overflows; the cast
   saturates / wraps per dtype semantics. We only verify the sampler
   doesn't crash. */
$a = NumPower::poisson([512], 200.0, 'uint8');
echo 'narrow_dtype_overflow: shape=', implode(',', $a->shape()), "\n";
?>
--EXPECT--
float4 0d: ndarray=OK shape=OK
float8 0d: ndarray=OK shape=OK
float16 0d: ndarray=OK shape=OK
float32 0d: ndarray=OK shape=OK
float64 0d: ndarray=OK shape=OK
float128 0d: ndarray=OK shape=OK
int8 0d: ndarray=OK shape=OK
uint8 0d: ndarray=OK shape=OK
int16 0d: ndarray=OK shape=OK
uint16 0d: ndarray=OK shape=OK
int32 0d: ndarray=OK shape=OK
uint32 0d: ndarray=OK shape=OK
int64 0d: ndarray=OK shape=OK
uint64 0d: ndarray=OK shape=OK
float4 zero: OK
float8 zero: OK
float16 zero: OK
float32 zero: OK
float64 zero: OK
float128 zero: OK
int8 zero: OK
uint8 zero: OK
int16 zero: OK
uint16 zero: OK
int32 zero: OK
uint32 zero: OK
int64 zero: OK
uint64 zero: OK
float4 one: OK
float8 one: OK
float16 one: OK
float32 one: OK
float64 one: OK
float128 one: OK
int8 one: OK
uint8 one: OK
int16 one: OK
uint16 one: OK
int32 one: OK
uint32 one: OK
int64 one: OK
uint64 one: OK
multidim: OK
shape_nd: OK
defaults: shape=8 dtype=OK device=cpu
scalar_shape: OK
knuth_edge_lam29.999: OK
ptrs_edge_lam30: OK
large_lam_no_hang: OK
tiny_lam_0.001: OK
int_lam_7: OK
string_lam_12.5: OK
narrow_dtype_overflow: shape=512
