--TEST--
NumPower::uniform() handles boundary shapes (0, 0-D, tiny, multi-dim) on every dtype
--FILE--
<?php
/* Boundary checks: 0-D shape (single scalar), 0-element shapes (empty
   per-dim), 1-element shapes, multi-dim, NDArray-as-shape, the default
   contract, and shapes that exceed the per-axis size of a small dtype.
   Each path goes through the same NDArray_Uniform dispatcher, so a
   small-shape regression there would crash before this test asserted
   anything. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

/* 0-D shape (size = 1, ndim = 0). The factory must return an NDArray
   (not collapse to a primitive zval) — same contract every other typed
   factory enforces. */
foreach ($dtypes as $dt) {
    if ($dt === 'uint64') { $low = '0'; $high = '100'; }
    elseif ($dt === 'float128') { $low = '0.0'; $high = '1.0'; }
    else { $low = 0; $high = 1; }
    $a = NumPower::uniform([], $low, $high, $dt);
    $is_ndarray = ($a instanceof NDArray);
    $shape_ok   = $a->shape() === [];
    echo $dt, ' 0d: ndarray=', ($is_ndarray ? 'OK' : 'BAD'),
         ' shape=', ($shape_ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [0] — zero-element 1-D. The buffer is allocated but never
   written; the typed-fill loop has to handle n == 0 cleanly. */
foreach ($dtypes as $dt) {
    if ($dt === 'uint64') { $low = '0'; $high = '100'; }
    elseif ($dt === 'float128') { $low = '0.0'; $high = '1.0'; }
    else { $low = 0; $high = 1; }
    $a = NumPower::uniform([0], $low, $high, $dt);
    $ok = $a->shape() === [0] && $a->size() === 0;
    echo $dt, ' zero: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Shape [1] — single-element 1-D. */
foreach ($dtypes as $dt) {
    if ($dt === 'uint64') { $low = '0'; $high = '100'; }
    elseif ($dt === 'float128') { $low = '0.0'; $high = '1.0'; }
    else { $low = 0; $high = 1; }
    $a = NumPower::uniform([1], $low, $high, $dt);
    $ok = $a->shape() === [1] && $a->size() === 1;
    echo $dt, ' one: ', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Multi-dim: 3-D shape. */
$a = NumPower::uniform([2, 3, 4], 0.0, 1.0, 'float32');
echo 'multidim: ', ($a->shape() === [2, 3, 4] && $a->size() === 24 ? 'OK' : 'BAD'), "\n";

/* Shape passed as an NDArray (ndarray_parse_typed_shape uses
   ZVAL_TO_NDARRAY so this branch is exercised). */
$shape_nd = new NDArray([4, 4]);
$a = NumPower::uniform($shape_nd, 0.0, 1.0, 'float32');
echo 'shape_nd: ', ($a->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";

/* Defaults — only the shape arg: low=0, high=1, dtype=float32, device=CPU. */
$a = NumPower::uniform([8]);
echo 'defaults: shape=', implode(',', $a->shape()),
     ' dtype=', (gettype($a[0]) === 'double' ? 'OK' : 'BAD'),
     ' device=', ($a->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Equal bounds: low == high. All samples reduce to low (no width). */
$a = NumPower::uniform([16], 7.5, 7.5, 'float32');
$arr = $a->toArray();
$all_same = true;
foreach ($arr as $v) { if ((float)$v !== 7.5) { $all_same = false; break; } }
echo 'equal_bounds: ', ($all_same ? 'OK' : 'BAD'), "\n";

/* Scalar shape: 1-D output sized by the scalar (ZVAL_TO_NDARRAY converts
   a bare long to a 1-element shape vector). */
$a = NumPower::uniform(5, 0.0, 1.0, 'float32');
$ok = $a->shape() === [5] && $a->size() === 5;
echo 'scalar_shape: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Large 1-D shape with strict range check — should never see exactly
   1.0 with the new sampler (legacy code allowed it). */
$n = 50000;
$a = NumPower::uniform([$n], 0.0, 1.0, 'float32');
$arr = $a->toArray();
$has_exactly_one = false;
$max = -INF;
foreach ($arr as $v) {
    $f = (float)$v;
    if ($f >= 1.0) { $has_exactly_one = true; }
    if ($f > $max) { $max = $f; }
}
echo 'strict_open_high: ', (!$has_exactly_one ? 'OK' : "BAD($max)"), "\n";

/* Tiny narrow-dtype range: shape [3, 3] on uint8 over [0, 4). */
$a = NumPower::uniform([3, 3], 0, 4, 'uint8');
$arr   = $a->toArray();
$range = true;
foreach ($arr as $row) {
    foreach ($row as $v) {
        if ($v < 0 || $v >= 4) { $range = false; break 2; }
    }
}
echo 'narrow_dtype_tight_range: ', ($range ? 'OK' : 'BAD'), "\n";
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
equal_bounds: OK
scalar_shape: OK
strict_open_high: OK
narrow_dtype_tight_range: OK
