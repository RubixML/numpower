--TEST--
NumPower::slice() works on GPU across every dtype; CPU and GPU produce identical results
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU slicing must:
   - produce the same values as CPU slicing for every dtype,
   - keep the result on the same device as the source,
   - cope with leading-axis (contiguous run), trailing-axis (strided pattern),
     and middle-axis (mixed) slicing equally well.
   For float128 the GPU stores values as double-double (hi, lo); the bytes that
   slice() copies must therefore travel through cudaMemcpy device→device
   unchanged so the dd round-trip is exact when we move the slice back to CPU. */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

function compare(NDArray $a, NDArray $b): bool {
    if ($a->shape() !== $b->shape()) return false;
    return $a->toArray() === $b->toArray();
}

/* Static NumPower::slice() keeps the source arrays alive across the many
   sub-tests below — the instance method now mutates $this and would
   destroy our CPU reference after the first call. */
foreach ($types as $t) {
    $src = new NDArray([[1,2,3,4],[5,6,7,8],[9,10,11,12]], $t);
    $gpu = $src->gpu();

    /* slice(1) — drop leading axis */
    $cpu_row = NumPower::slice($src, 1);
    $gpu_row = NumPower::slice($gpu, 1);
    $ok = compare($cpu_row, $gpu_row->cpu());
    echo "$t slice(1): ", $ok ? "OK" : "BAD", "\n";

    /* slice([], -1) — last column (non-contiguous source-memory pattern) */
    $cpu_col = NumPower::slice($src, [], -1);
    $gpu_col = NumPower::slice($gpu, [], -1);
    $ok = compare($cpu_col, $gpu_col->cpu());
    echo "$t slice([],-1): ", $ok ? "OK" : "BAD", "\n";

    /* slice([0, 2], [1, 4, 2]) — range + step on both axes */
    $cpu_sub = NumPower::slice($src, [0, 2], [1, 4, 2]);
    $gpu_sub = NumPower::slice($gpu, [0, 2], [1, 4, 2]);
    $ok = compare($cpu_sub, $gpu_sub->cpu());
    echo "$t slice([0,2],[1,4,2]): ", $ok ? "OK" : "BAD", "\n";

    /* slice(0, 0) — 0-D scalar from GPU */
    $cpu_s = NumPower::slice($src, 0, 0);
    $gpu_s = NumPower::slice($gpu, 0, 0);
    $ok = ($cpu_s === $gpu_s);
    echo "$t slice(0,0) scalar: ", $ok ? "OK" : "BAD cpu=" . var_export($cpu_s, true) . " gpu=" . var_export($gpu_s, true), "\n";

    /* slice on GPU stays on GPU when result is not 0-D */
    $on_gpu = NumPower::slice($gpu, [0, 2]);
    echo "$t slice on gpu->isGPU: ", $on_gpu->isGPU() ? "OK" : "BAD", "\n";
}

/* 3-D GPU slice covering all three positional reductions. */
$cube = new NDArray([[[1,2],[3,4]],[[5,6],[7,8]]], 'float32');
$gpu  = $cube->gpu();
echo "cube slice(1) parity: ", compare(NumPower::slice($cube, 1), NumPower::slice($gpu, 1)->cpu()) ? "OK" : "BAD", "\n";
echo "cube slice([],1) parity: ", compare(NumPower::slice($cube, [], 1), NumPower::slice($gpu, [], 1)->cpu()) ? "OK" : "BAD", "\n";
echo "cube slice([],[],1) parity: ", compare(NumPower::slice($cube, [], [], 1), NumPower::slice($gpu, [], [], 1)->cpu()) ? "OK" : "BAD", "\n";

/* Negative step on GPU. */
$g = NumPower::arange(8.0)->gpu();
echo "gpu negstep parity: ",
     (NumPower::slice($g, [7, 0, -2])->cpu()->toArray() === [7.0, 5.0, 3.0, 1.0] ? "OK" : "BAD"), "\n";
?>
--EXPECT--
float4 slice(1): OK
float4 slice([],-1): OK
float4 slice([0,2],[1,4,2]): OK
float4 slice(0,0) scalar: OK
float4 slice on gpu->isGPU: OK
float8 slice(1): OK
float8 slice([],-1): OK
float8 slice([0,2],[1,4,2]): OK
float8 slice(0,0) scalar: OK
float8 slice on gpu->isGPU: OK
float16 slice(1): OK
float16 slice([],-1): OK
float16 slice([0,2],[1,4,2]): OK
float16 slice(0,0) scalar: OK
float16 slice on gpu->isGPU: OK
float32 slice(1): OK
float32 slice([],-1): OK
float32 slice([0,2],[1,4,2]): OK
float32 slice(0,0) scalar: OK
float32 slice on gpu->isGPU: OK
float64 slice(1): OK
float64 slice([],-1): OK
float64 slice([0,2],[1,4,2]): OK
float64 slice(0,0) scalar: OK
float64 slice on gpu->isGPU: OK
float128 slice(1): OK
float128 slice([],-1): OK
float128 slice([0,2],[1,4,2]): OK
float128 slice(0,0) scalar: OK
float128 slice on gpu->isGPU: OK
int8 slice(1): OK
int8 slice([],-1): OK
int8 slice([0,2],[1,4,2]): OK
int8 slice(0,0) scalar: OK
int8 slice on gpu->isGPU: OK
uint8 slice(1): OK
uint8 slice([],-1): OK
uint8 slice([0,2],[1,4,2]): OK
uint8 slice(0,0) scalar: OK
uint8 slice on gpu->isGPU: OK
int16 slice(1): OK
int16 slice([],-1): OK
int16 slice([0,2],[1,4,2]): OK
int16 slice(0,0) scalar: OK
int16 slice on gpu->isGPU: OK
uint16 slice(1): OK
uint16 slice([],-1): OK
uint16 slice([0,2],[1,4,2]): OK
uint16 slice(0,0) scalar: OK
uint16 slice on gpu->isGPU: OK
int32 slice(1): OK
int32 slice([],-1): OK
int32 slice([0,2],[1,4,2]): OK
int32 slice(0,0) scalar: OK
int32 slice on gpu->isGPU: OK
uint32 slice(1): OK
uint32 slice([],-1): OK
uint32 slice([0,2],[1,4,2]): OK
uint32 slice(0,0) scalar: OK
uint32 slice on gpu->isGPU: OK
int64 slice(1): OK
int64 slice([],-1): OK
int64 slice([0,2],[1,4,2]): OK
int64 slice(0,0) scalar: OK
int64 slice on gpu->isGPU: OK
uint64 slice(1): OK
uint64 slice([],-1): OK
uint64 slice([0,2],[1,4,2]): OK
uint64 slice(0,0) scalar: OK
uint64 slice on gpu->isGPU: OK
cube slice(1) parity: OK
cube slice([],1) parity: OK
cube slice([],[],1) parity: OK
gpu negstep parity: OK
