--TEST--
NumPower::diag() — both directions cover every dtype on CPU
--FILE--
<?php
/* diag has dual semantics:
    - 1-D input → 2-D N×N matrix with the input on the main diagonal.
    - 2-D input → 1-D vector of min(rows, cols) elements (the diagonal).
   Both directions must respect the explicit output dtype and produce
   exactly the values the dtype encodes.

   The old implementation was float32 / CPU only — would have written
   `float` 1.0 / read `sizeof(float)` per element regardless of the
   actual dtype, mangling any non-float32 input. This test pins the
   per-dtype correctness for both directions. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

/* 1-D → 2-D: trace == sum(input). */
foreach ($dtypes as $dt) {
    $src = NumPower::arange(4, 0, 1, $dt);   /* [0, 1, 2, 3] in $dt */
    $m   = NumPower::diag($src, $dt);
    $shape_ok = $m->shape() === [4, 4];
    $trace_ok = (string)NumPower::sum($m) === '6';  /* 0+1+2+3 */
    $off_ok   = (string)$m[0][1] === '0' || (string)$m[0][1] === '0.0';
    echo "$dt 1d→2d: shape=", ($shape_ok ? 'OK' : 'BAD'),
         " trace=", ($trace_ok ? 'OK' : 'BAD'),
         " off=",   ($off_ok   ? 'OK' : 'BAD'), "\n";
}

/* 2-D → 1-D: trace of identity(N, dtype) is N. */
foreach ($dtypes as $dt) {
    $eye = NumPower::identity(5, $dt);
    $v   = NumPower::diag($eye, $dt);
    $shape_ok = $v->shape() === [5];
    $sum_ok   = (string)NumPower::sum($v) === '5';
    echo "$dt 2d→1d: shape=", ($shape_ok ? 'OK' : 'BAD'),
         " sum=", ($sum_ok ? 'OK' : 'BAD'), "\n";
}

/* Non-square: 3×4 → length 3. */
$src = NumPower::array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 'int32');
$d   = NumPower::diag($src, 'int32');
echo "3x4 diag: shape=", json_encode($d->shape()),
     " vals=", (string)$d, "\n";

/* Non-square: 4×3 → length 3. */
$src = NumPower::array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 'int32');
$d   = NumPower::diag($src, 'int32');
echo "4x3 diag: shape=", json_encode($d->shape()),
     " vals=", (string)$d, "\n";

/* 1-D fp128 string round-trip */
$src = NumPower::array(['1.5', '2.5', '3.5'], 'float128');
$m   = NumPower::diag($src, 'float128');
echo "fp128 1d→2d trace=", (string)NumPower::sum($m), "\n";

/* uint64 string round-trip */
$src = NumPower::array(['18446744073709551610', '18446744073709551612'], 'uint64');
$m   = NumPower::diag($src, 'uint64');
echo "uint64 max diag[0][0]=", (string)$m[0][0], "\n";
echo "uint64 max diag[1][1]=", (string)$m[1][1], "\n";

/* Cast on construction: float64 input → int32 output */
$src = NumPower::array([1.5, 2.5, 3.5], 'float64');
$m   = NumPower::diag($src, 'int32');
echo "cast f64→i32 1d→2d trace=", (string)NumPower::sum($m), "\n";
?>
--EXPECT--
float4 1d→2d: shape=OK trace=OK off=OK
float8 1d→2d: shape=OK trace=OK off=OK
float16 1d→2d: shape=OK trace=OK off=OK
float32 1d→2d: shape=OK trace=OK off=OK
float64 1d→2d: shape=OK trace=OK off=OK
float128 1d→2d: shape=OK trace=OK off=OK
int8 1d→2d: shape=OK trace=OK off=OK
uint8 1d→2d: shape=OK trace=OK off=OK
int16 1d→2d: shape=OK trace=OK off=OK
uint16 1d→2d: shape=OK trace=OK off=OK
int32 1d→2d: shape=OK trace=OK off=OK
uint32 1d→2d: shape=OK trace=OK off=OK
int64 1d→2d: shape=OK trace=OK off=OK
uint64 1d→2d: shape=OK trace=OK off=OK
float4 2d→1d: shape=OK sum=OK
float8 2d→1d: shape=OK sum=OK
float16 2d→1d: shape=OK sum=OK
float32 2d→1d: shape=OK sum=OK
float64 2d→1d: shape=OK sum=OK
float128 2d→1d: shape=OK sum=OK
int8 2d→1d: shape=OK sum=OK
uint8 2d→1d: shape=OK sum=OK
int16 2d→1d: shape=OK sum=OK
uint16 2d→1d: shape=OK sum=OK
int32 2d→1d: shape=OK sum=OK
uint32 2d→1d: shape=OK sum=OK
int64 2d→1d: shape=OK sum=OK
uint64 2d→1d: shape=OK sum=OK
3x4 diag: shape=[3] vals=[1, 6, 11]
4x3 diag: shape=[3] vals=[1, 5, 9]
fp128 1d→2d trace=7.5
uint64 max diag[0][0]=18446744073709551610
uint64 max diag[1][1]=18446744073709551612
cast f64→i32 1d→2d trace=6
