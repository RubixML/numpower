--TEST--
NumPower::truncatedNormal() handles boundary shapes (0, 0-D, tiny, multi-dim) on every dtype
--FILE--
<?php
/* Boundary checks parallel to the normal() edge-case suite plus the
   defining truncation property: every value in the output must lie in
   [loc - 2σ, loc + 2σ]. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

mt_srand(42);
srand(42);

/* 0-D shape (size = 1, ndim = 0). The factory must return an NDArray
   (not a primitive zval) — same contract every other typed factory
   enforces. */
foreach ($dtypes as $dt) {
    $a = NumPower::truncatedNormal([], 0.0, 1.0, $dt);
    echo $dt, ' 0d: ndarray=', ($a instanceof NDArray ? 'OK' : 'BAD'),
         ' shape=', ($a->shape() === [] ? 'OK' : 'BAD'),
         "\n";
}

/* Shape [0] — zero-element 1-D. */
foreach ($dtypes as $dt) {
    $a = NumPower::truncatedNormal([0], 0.0, 1.0, $dt);
    echo $dt, ' zero: ', ($a->shape() === [0] && $a->size() === 0 ? 'OK' : 'BAD'), "\n";
}

/* Shape [1] — single-element 1-D. */
foreach ($dtypes as $dt) {
    $a = NumPower::truncatedNormal([1], 0.0, 1.0, $dt);
    echo $dt, ' one: ', ($a->shape() === [1] && $a->size() === 1 ? 'OK' : 'BAD'), "\n";
}

/* Multi-dim. */
$a = NumPower::truncatedNormal([2, 3, 4], 0.0, 1.0, 'float32');
echo 'multidim: ', ($a->shape() === [2, 3, 4] && $a->size() === 24 ? 'OK' : 'BAD'), "\n";

/* Shape as NDArray. */
$shape_nd = new NDArray([4, 4]);
$a = NumPower::truncatedNormal($shape_nd, 0.0, 1.0, 'float32');
echo 'shape_nd: ', ($a->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";

/* Defaults: shape only. */
$a = NumPower::truncatedNormal([8]);
echo 'defaults: shape=', implode(',', $a->shape()),
     ' dtype=', (gettype($a[0]) === 'double' ? 'OK' : 'BAD'),
     ' device=', ($a->isGPU() ? 'gpu' : 'cpu'), "\n";

/* Scale=0 — every sample reduces to loc. */
$a = NumPower::truncatedNormal([8], 7.0, 0.0, 'float32');
$arr = $a->toArray();
$all_loc = true;
foreach ($arr as $v) { if ((float)$v != 7.0) $all_loc = false; }
echo 'scale_zero: ', ($all_loc ? 'OK' : 'BAD'), "\n";

/* Large shape with narrow dtype — uint8 range is [0, 255]. With
   loc=200, scale=30 the truncation window is [140, 260] which falls
   partly outside uint8; values above 255 wrap to (value - 256) per
   unsigned cast — but the truncation check on z is exact, so values
   are still drawn from [loc-2σ, loc+2σ] before casting. */
$a = NumPower::truncatedNormal([512], 200, 30, 'uint8');
echo 'narrow_dtype_wide_dist: shape=', ($a->shape() === [512] ? 'OK' : 'BAD'), "\n";

/* Truncation window check — every sample must lie in [-2, 2] for std
   N(0, 1). Multiple independent runs to make sure the kernel never
   leaks a non-truncated sample. */
$total_out = 0;
for ($run = 0; $run < 5; $run++) {
    $a = NumPower::truncatedNormal([4096], 0.0, 1.0, 'float32');
    foreach ($a->toArray() as $v) {
        $f = (float)$v;
        if ($f < -2.0 - 1e-5 || $f > 2.0 + 1e-5) $total_out++;
    }
}
echo 'window_repeated: total_out_of_window=', $total_out, "\n";

/* Repeatability of mean across runs — should NOT collapse to a single
   value (previous bug: pinned seed produced identical streams). */
$means = [];
for ($run = 0; $run < 5; $run++) {
    $a = NumPower::truncatedNormal([2048], 0.0, 1.0, 'float32');
    $means[] = (float) NumPower::mean($a);
}
$unique_means = count(array_unique(array_map(fn($x) => round($x, 6), $means)));
echo 'distinct_means_across_runs: ', ($unique_means >= 4 ? 'OK' : "BAD($unique_means)"), "\n";
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
scale_zero: OK
narrow_dtype_wide_dist: shape=OK
window_repeated: total_out_of_window=0
distinct_means_across_runs: OK
