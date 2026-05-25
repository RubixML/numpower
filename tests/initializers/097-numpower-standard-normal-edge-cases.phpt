--TEST--
NumPower::standardNormal() handles boundary shapes (0, 0-D, tiny, multi-dim) on every dtype
--FILE--
<?php
/* Boundary checks: 0-D shape (single scalar), 0-element shapes (empty
   per-dim), 1-element shapes, multi-dim, NDArray-as-shape, the default
   contract, and shapes that exceed the per-axis size of a small dtype.
   Each path goes through the same NDArray_Normal dispatcher as
   `normal()`, but standardNormal has its own defaults so the
   per-element fill at n == 0 / n == 1 / large n needs its own coverage. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

/* 0-D shape (size = 1, ndim = 0). The factory must return an NDArray
   (not collapse to a primitive zval) — same contract every other typed
   factory enforces. */
foreach ($dtypes as $dt) {
    $a = NumPower::standardNormal([], $dt);
    $is_ndarray = ($a instanceof NDArray);
    $shape_ok   = $a->shape() === [];
    echo $dt, ' 0d: ndarray=', ($is_ndarray ? 'OK' : 'BAD'),
         ' shape=', ($shape_ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [0] — zero-element 1-D. The buffer is allocated but never
   written; the typed-fill loop has to handle n == 0 cleanly. */
foreach ($dtypes as $dt) {
    $a = NumPower::standardNormal([0], $dt);
    $ok = $a->shape() === [0] && $a->size() === 0;
    echo $dt, ' zero: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [1] — single-element 1-D. The CPU Box-Muller sampler returns
   two samples per polar draw; the single-element case must write only
   one. */
foreach ($dtypes as $dt) {
    $a = NumPower::standardNormal([1], $dt);
    $ok = $a->shape() === [1] && $a->size() === 1;
    echo $dt, ' one: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Multi-dim: 3-D shape. */
$a = NumPower::standardNormal([2, 3, 4], 'float32');
echo 'multidim: ', ($a->shape() === [2, 3, 4] && $a->size() === 24 ? 'OK' : 'BAD'), "\n";

/* Shape passed as an NDArray (ndarray_parse_typed_shape uses
   ZVAL_TO_NDARRAY so this branch is exercised). */
$shape_nd = new NDArray([4, 4]);
$a = NumPower::standardNormal($shape_nd, 'float32');
echo 'shape_nd: ', ($a->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";

/* Defaults — only the shape arg, dtype=float32, device=CPU. */
$a = NumPower::standardNormal([8]);
echo 'defaults: shape=', implode(',', $a->shape()),
     ' dtype=', (gettype($a[0]) === 'double' ? 'OK' : 'BAD'),
     ' device=', ($a->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Large 1-D shape — ensures the fill loop handles a buffer larger than
   any CPU cache line / cuRAND batch boundary cleanly. */
$a = NumPower::standardNormal([4096], 'float32');
$ok = $a->shape() === [4096] && $a->size() === 4096;
echo 'large_1d: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Shape requested via a primitive scalar — ZVAL_TO_NDARRAY converts a
   bare long to a 1-D shape vector [long]. We end up with a 1-D output
   sized by that scalar. */
$a = NumPower::standardNormal(5, 'float32');
$ok = $a->shape() === [5] && $a->size() === 5;
echo 'scalar_shape: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Tiny per-axis size on a narrow dtype: shape [3, 3] on uint8. The
   quantisation collapses ~68% of the |z| < 1 mass to 0 and ~16% each
   to ±1 (which become 1 and UINT8_MAX); the test only verifies the
   sampler doesn't crash at small dims. */
$a = NumPower::standardNormal([3, 3], 'uint8');
$ok = $a->shape() === [3, 3];
echo 'narrow_dtype_small_shape: ', ($ok ? 'OK' : 'BAD'), "\n";
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
large_1d: OK
scalar_shape: OK
narrow_dtype_small_shape: OK
