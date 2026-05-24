--TEST--
NumPower::normal() handles boundary shapes (0, 0-D, tiny, multi-dim) on every dtype
--FILE--
<?php
/* Boundary checks: 0-D shape (single scalar), 0-element shapes (empty
   per-dim), 1-element shapes, multi-dim, and shapes that exceed the
   per-axis size of a small dtype's representable range — making sure
   the per-element fill loop is correct at every n. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

/* 0-D shape (size = 1, ndim = 0). The factory must return an NDArray
   (not collapse to a primitive zval) — same contract every other typed
   factory enforces. */
foreach ($dtypes as $dt) {
    $a = NumPower::normal([], 0.0, 1.0, $dt);
    $is_ndarray = ($a instanceof NDArray);
    $shape_ok   = $a->shape() === [];
    echo $dt, ' 0d: ndarray=', ($is_ndarray ? 'OK' : 'BAD'),
         ' shape=', ($shape_ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [0] — zero-element 1-D. The buffer is allocated but never
   written; the typed-fill loop has to handle n == 0 cleanly. */
foreach ($dtypes as $dt) {
    $a = NumPower::normal([0], 0.0, 1.0, $dt);
    $ok = $a->shape() === [0] && $a->size() === 0;
    echo $dt, ' zero: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [1] — single-element 1-D. The cpu Box-Muller path is paired
   (returns two samples per polar draw); we have to make sure the
   single-element case writes only one. */
foreach ($dtypes as $dt) {
    $a = NumPower::normal([1], 0.0, 1.0, $dt);
    $ok = $a->shape() === [1] && $a->size() === 1;
    echo $dt, ' one: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Multi-dim: 3-D shape. */
$a = NumPower::normal([2, 3, 4], 0.0, 1.0, 'float32');
echo 'multidim: ', ($a->shape() === [2, 3, 4] && $a->size() === 24 ? 'OK' : 'BAD'), "\n";

/* Shape passed as an NDArray (ndarray_parse_typed_shape uses
   ZVAL_TO_NDARRAY so this branch is exercised). */
$shape_nd = new NDArray([4, 4]);
$a = NumPower::normal($shape_nd, 0.0, 1.0, 'float32');
echo 'shape_nd: ', ($a->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";

/* Defaults — only the shape arg, loc=0, scale=1, dtype=float32,
   device=CPU. */
$a = NumPower::normal([8]);
echo 'defaults: shape=', implode(',', $a->shape()),
     ' dtype=', (gettype($a[0]) === 'double' ? 'OK' : 'BAD'),
     ' device=', ($a->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Scale=0 — every sample reduces to loc. */
$a = NumPower::normal([8], 7.0, 0.0, 'float32');
$mean = NumPower::mean($a);
$std  = NumPower::std($a);
echo 'scale_zero: mean=', ($mean == 7.0 ? 'OK' : "BAD($mean)"),
     ' std=',  ($std  == 0.0 ? 'OK' : "BAD($std)"),  "\n";

/* Large shape that exceeds a narrow dtype's representable range
   (uint8 range is [0, 255]; we ask for samples around 200 ± 30 which
   may sometimes wrap into the [0, 255] window; the test only verifies
   the sampler doesn't crash on the boundary). */
$a = NumPower::normal([512], 200, 30, 'uint8');
$shape_ok = $a->shape() === [512];
echo 'narrow_dtype_wide_dist: ', ($shape_ok ? 'OK' : 'BAD'), "\n";
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
scale_zero: mean=OK std=OK
narrow_dtype_wide_dist: OK
