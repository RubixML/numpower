--TEST--
NDArray::fill() works on GPU across all dtypes (CPU/GPU parity)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Filling a GPU NDArray writes through a CPU staging buffer + TypedH2D —
   for float128 the staging buffer holds __float128/dd and TypedH2D converts
   it to the on-device (hi, lo) layout. Confirm the post-fill values on every
   dtype match what the CPU path produces. Values chosen to be representable
   in every dtype. */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($types as $t) {
    /* String fill */
    $cpu = new NDArray([1,2,3,4], $t);
    $cpu->fill('3');

    $gpu = (new NDArray([1,2,3,4], $t))->gpu();
    $gpu->fill('3');
    $back = $gpu->cpu();
    echo "$t fill('3') CPU==GPU: ",
         ($cpu->toArray() === $back->toArray() ? "OK" : "BAD"), "\n";

    /* Int fill */
    $cpu = new NDArray([1,2,3,4], $t);
    $cpu->fill(1);
    $gpu = (new NDArray([1,2,3,4], $t))->gpu();
    $gpu->fill(1);
    $back = $gpu->cpu();
    echo "$t fill(1) CPU==GPU: ",
         ($cpu->toArray() === $back->toArray() ? "OK" : "BAD"), "\n";

    /* Float fill */
    $cpu = new NDArray([1,2,3,4], $t);
    $cpu->fill(2.0);
    $gpu = (new NDArray([1,2,3,4], $t))->gpu();
    $gpu->fill(2.0);
    $back = $gpu->cpu();
    echo "$t fill(2.0) CPU==GPU: ",
         ($cpu->toArray() === $back->toArray() ? "OK" : "BAD"), "\n";

    /* True / false */
    $cpu = new NDArray([1,2,3,4], $t);
    $cpu->fill(true);
    $gpu = (new NDArray([1,2,3,4], $t))->gpu();
    $gpu->fill(true);
    $back = $gpu->cpu();
    echo "$t fill(true) CPU==GPU: ",
         ($cpu->toArray() === $back->toArray() ? "OK" : "BAD"), "\n";

    $cpu = new NDArray([1,2,3,4], $t);
    $cpu->fill(false);
    $gpu = (new NDArray([1,2,3,4], $t))->gpu();
    $gpu->fill(false);
    $back = $gpu->cpu();
    echo "$t fill(false) CPU==GPU: ",
         ($cpu->toArray() === $back->toArray() ? "OK" : "BAD"), "\n";
}

/* int64 PHP_INT_MAX must survive the GPU round-trip — the staging buffer
   stays in int64 so no float64 mantissa loss. */
$g = (new NDArray([0,0,0], 'int64'))->gpu();
$g->fill(PHP_INT_MAX);
$back = $g->cpu();
echo "int64 GPU PHP_INT_MAX: ",
     ($back->toArray() === [PHP_INT_MAX, PHP_INT_MAX, PHP_INT_MAX] ? "OK" : "BAD"), "\n";

/* uint64 string fill on GPU */
$g = (new NDArray(['0','0','0'], 'uint64'))->gpu();
$g->fill('18446744073709551615');
$back = $g->cpu();
echo "uint64 GPU max: ",
     ($back->toArray() === ['18446744073709551615','18446744073709551615','18446744073709551615'] ? "OK" : "BAD"), "\n";

/* float128 high-precision string fill on GPU. The GPU stores fp128 as a
   double-double pair, so we expect ~32 digits — about one ulp slack vs the
   CPU __float128 path. Assert "starts with" the high prefix. */
$g = (new NDArray(['0','0','0'], 'float128'))->gpu();
$g->fill('3.14159265358979323846264338327950288');
$back = $g->cpu();
$prefix = '3.141592653589793238462643383279';
foreach ($back->toArray() as $i => $v) {
    echo "fp128 GPU[$i] starts: ", (str_starts_with($v, $prefix) ? "OK" : "BAD got=$v"), "\n";
}
?>
--EXPECT--
float4 fill('3') CPU==GPU: OK
float4 fill(1) CPU==GPU: OK
float4 fill(2.0) CPU==GPU: OK
float4 fill(true) CPU==GPU: OK
float4 fill(false) CPU==GPU: OK
float8 fill('3') CPU==GPU: OK
float8 fill(1) CPU==GPU: OK
float8 fill(2.0) CPU==GPU: OK
float8 fill(true) CPU==GPU: OK
float8 fill(false) CPU==GPU: OK
float16 fill('3') CPU==GPU: OK
float16 fill(1) CPU==GPU: OK
float16 fill(2.0) CPU==GPU: OK
float16 fill(true) CPU==GPU: OK
float16 fill(false) CPU==GPU: OK
float32 fill('3') CPU==GPU: OK
float32 fill(1) CPU==GPU: OK
float32 fill(2.0) CPU==GPU: OK
float32 fill(true) CPU==GPU: OK
float32 fill(false) CPU==GPU: OK
float64 fill('3') CPU==GPU: OK
float64 fill(1) CPU==GPU: OK
float64 fill(2.0) CPU==GPU: OK
float64 fill(true) CPU==GPU: OK
float64 fill(false) CPU==GPU: OK
float128 fill('3') CPU==GPU: OK
float128 fill(1) CPU==GPU: OK
float128 fill(2.0) CPU==GPU: OK
float128 fill(true) CPU==GPU: OK
float128 fill(false) CPU==GPU: OK
int8 fill('3') CPU==GPU: OK
int8 fill(1) CPU==GPU: OK
int8 fill(2.0) CPU==GPU: OK
int8 fill(true) CPU==GPU: OK
int8 fill(false) CPU==GPU: OK
uint8 fill('3') CPU==GPU: OK
uint8 fill(1) CPU==GPU: OK
uint8 fill(2.0) CPU==GPU: OK
uint8 fill(true) CPU==GPU: OK
uint8 fill(false) CPU==GPU: OK
int16 fill('3') CPU==GPU: OK
int16 fill(1) CPU==GPU: OK
int16 fill(2.0) CPU==GPU: OK
int16 fill(true) CPU==GPU: OK
int16 fill(false) CPU==GPU: OK
uint16 fill('3') CPU==GPU: OK
uint16 fill(1) CPU==GPU: OK
uint16 fill(2.0) CPU==GPU: OK
uint16 fill(true) CPU==GPU: OK
uint16 fill(false) CPU==GPU: OK
int32 fill('3') CPU==GPU: OK
int32 fill(1) CPU==GPU: OK
int32 fill(2.0) CPU==GPU: OK
int32 fill(true) CPU==GPU: OK
int32 fill(false) CPU==GPU: OK
uint32 fill('3') CPU==GPU: OK
uint32 fill(1) CPU==GPU: OK
uint32 fill(2.0) CPU==GPU: OK
uint32 fill(true) CPU==GPU: OK
uint32 fill(false) CPU==GPU: OK
int64 fill('3') CPU==GPU: OK
int64 fill(1) CPU==GPU: OK
int64 fill(2.0) CPU==GPU: OK
int64 fill(true) CPU==GPU: OK
int64 fill(false) CPU==GPU: OK
uint64 fill('3') CPU==GPU: OK
uint64 fill(1) CPU==GPU: OK
uint64 fill(2.0) CPU==GPU: OK
uint64 fill(true) CPU==GPU: OK
uint64 fill(false) CPU==GPU: OK
int64 GPU PHP_INT_MAX: OK
uint64 GPU max: OK
fp128 GPU[0] starts: OK
fp128 GPU[1] starts: OK
fp128 GPU[2] starts: OK
