--TEST--
NumPower::randomBinomial() handles boundary shapes and (n, p) values
--FILE--
<?php
/* Boundary checks: 0-D shape, [0] shape, [1] shape, multi-dim,
   NDArray-as-shape, the default contract, and degenerate distribution
   parameters. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

/* 0-D shape — must return an NDArray, not a primitive zval. */
foreach ($dtypes as $dt) {
    $a = NumPower::randomBinomial([], 10, 0.5, $dt);
    $is_ndarray = ($a instanceof NDArray);
    $shape_ok   = $a->shape() === [];
    echo $dt, ' 0d: ndarray=', ($is_ndarray ? 'OK' : 'BAD'),
         ' shape=', ($shape_ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [0] — zero-element 1-D. */
foreach ($dtypes as $dt) {
    $a = NumPower::randomBinomial([0], 10, 0.5, $dt);
    $ok = $a->shape() === [0] && $a->size() === 0;
    echo $dt, ' zero: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [1]. */
foreach ($dtypes as $dt) {
    $a = NumPower::randomBinomial([1], 10, 0.5, $dt);
    $ok = $a->shape() === [1] && $a->size() === 1;
    echo $dt, ' one: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Multi-dim. */
$a = NumPower::randomBinomial([2, 3, 4], 10, 0.5, 'float32');
echo 'multidim: ', ($a->shape() === [2, 3, 4] && $a->size() === 24 ? 'OK' : 'BAD'), "\n";

/* Shape passed as an NDArray. */
$shape_nd = new NDArray([4, 4]);
$a = NumPower::randomBinomial($shape_nd, 10, 0.5, 'float32');
echo 'shape_nd: ', ($a->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";

/* Defaults — only required args: n=20, p=0.3 → dtype=float32, device=CPU. */
$a = NumPower::randomBinomial([8], 20, 0.3);
echo 'defaults: shape=', implode(',', $a->shape()),
     ' dtype=', (gettype($a[0]) === 'double' ? 'OK' : 'BAD'),
     ' device=', ($a->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Scalar shape via ZVAL_TO_NDARRAY. */
$a = NumPower::randomBinomial(5, 10, 0.5, 'float32');
$ok = $a->shape() === [5] && $a->size() === 5;
echo 'scalar_shape: ', ($ok ? 'OK' : 'BAD'), "\n";

/* p = 1.0 — every sample equals n. */
$a = NumPower::randomBinomial([16], 7, 1.0, 'int32');
$all_n = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 7) { $all_n = false; break; }
}
echo 'p_one_all_n: ', ($all_n ? 'OK' : 'BAD'), "\n";

/* p = 0.0 — every sample equals 0. */
$a = NumPower::randomBinomial([16], 7, 0.0, 'int32');
$all_zero = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'p_zero_all_zero: ', ($all_zero ? 'OK' : 'BAD'), "\n";

/* n = 1 (Bernoulli) — every sample is 0 or 1. */
$a = NumPower::randomBinomial([1024], 1, 0.5, 'int32');
$bernoulli_ok = true;
foreach ($a->toArray() as $v) {
    $iv = (int)$v;
    if ($iv !== 0 && $iv !== 1) { $bernoulli_ok = false; break; }
}
echo 'n_one_bernoulli: ', ($bernoulli_ok ? 'OK' : 'BAD'), "\n";

/* n = 0 with p = 1.0 — still all zeros (degenerate distribution). */
$a = NumPower::randomBinomial([16], 0, 1.0, 'int32');
$all_zero = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 0) { $all_zero = false; break; }
}
echo 'n_zero_p_one: ', ($all_zero ? 'OK' : 'BAD'), "\n";

/* Integer-typed n passed (legacy contract uses float). */
$a = NumPower::randomBinomial([64], 10.0, 0.5, 'int32');
$arr = $a->toArray();
$mean = array_sum($arr) / count($arr);
echo 'float_n_works: ', (abs($mean - 5.0) < 1.5 ? 'OK' : "BAD($mean)"), "\n";

/* Narrow dtype + n at the dtype's representable edge. uint8 holds 255;
   B(255, 1.0) saturates uint8 exactly. */
$a = NumPower::randomBinomial([16], 255, 1.0, 'uint8');
$ok = true;
foreach ($a->toArray() as $v) {
    if ((int)$v !== 255) { $ok = false; break; }
}
echo 'uint8_max_saturation: ', ($ok ? 'OK' : 'BAD'), "\n";
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
p_one_all_n: OK
p_zero_all_zero: OK
n_one_bernoulli: OK
n_zero_p_one: OK
float_n_works: OK
uint8_max_saturation: OK
