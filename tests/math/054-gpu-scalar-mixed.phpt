--TEST--
Scalar OP GPU and GPU OP scalar work for every dtype (scalar stays on CPU, result on GPU). Real-array cross-device still throws.
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Verify the scalar+GPU and GPU+scalar operator dispatch:
   - returns a GPU array (except for float128 where GPU has no kernel)
   - returns the correct values
   - leaves real-array cross-device cases throwing */

$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($dtypes as $dt) {
    $a = NumPower::array([4, 4, 4, 4], $dt)->gpu();

    /* GPU + scalar — every dtype now keeps result on GPU. float128 uses
       double-double emulation on GPU. */
    $r1 = $a + 2;
    $expect_gpu = 1;
    $cpu1 = $r1->isGPU() ? $r1->cpu() : $r1;
    echo "$dt gpu+scalar isGPU=", $r1->isGPU(), " (want=$expect_gpu) v=", (string)$cpu1[0], "\n";

    /* scalar + GPU */
    $r2 = 2 + $a;
    $cpu2 = $r2->isGPU() ? $r2->cpu() : $r2;
    echo "$dt scalar+gpu isGPU=", $r2->isGPU(), " (want=$expect_gpu) v=", (string)$cpu2[0], "\n";

    /* GPU * scalar */
    $r3 = $a * 2;
    $cpu3 = $r3->isGPU() ? $r3->cpu() : $r3;
    echo "$dt gpu*scalar isGPU=", $r3->isGPU(), " v=", (string)$cpu3[0], "\n";

    /* GPU / scalar (integer dtypes always promote to float for "/") */
    $r4 = $a / 2;
    $cpu4 = $r4->isGPU() ? $r4->cpu() : $r4;
    echo "$dt gpu/scalar isGPU=", $r4->isGPU(), " v=", (string)$cpu4[0], "\n";
}

/* Real arr CPU + real arr GPU still throws */
$cpu = NumPower::array([1, 2, 3, 4], 'float32');
$gpu = $cpu->gpu();
try {
    $bad = $cpu + $gpu;
    echo "FAIL: cross-device array+array didn't throw\n";
} catch (Error $e) {
    echo "cross-device throws: ", $e->getMessage(), "\n";
}

try {
    $bad = $gpu + $cpu;
    echo "FAIL: cross-device array+array didn't throw\n";
} catch (Error $e) {
    echo "cross-device throws: ", $e->getMessage(), "\n";
}
?>
--EXPECT--
float4 gpu+scalar isGPU=1 (want=1) v=6
float4 scalar+gpu isGPU=1 (want=1) v=6
float4 gpu*scalar isGPU=1 v=6
float4 gpu/scalar isGPU=1 v=2
float8 gpu+scalar isGPU=1 (want=1) v=6
float8 scalar+gpu isGPU=1 (want=1) v=6
float8 gpu*scalar isGPU=1 v=8
float8 gpu/scalar isGPU=1 v=2
float16 gpu+scalar isGPU=1 (want=1) v=6
float16 scalar+gpu isGPU=1 (want=1) v=6
float16 gpu*scalar isGPU=1 v=8
float16 gpu/scalar isGPU=1 v=2
float32 gpu+scalar isGPU=1 (want=1) v=6
float32 scalar+gpu isGPU=1 (want=1) v=6
float32 gpu*scalar isGPU=1 v=8
float32 gpu/scalar isGPU=1 v=2
float64 gpu+scalar isGPU=1 (want=1) v=6
float64 scalar+gpu isGPU=1 (want=1) v=6
float64 gpu*scalar isGPU=1 v=8
float64 gpu/scalar isGPU=1 v=2
float128 gpu+scalar isGPU=1 (want=1) v=6
float128 scalar+gpu isGPU=1 (want=1) v=6
float128 gpu*scalar isGPU=1 v=8
float128 gpu/scalar isGPU=1 v=2
int8 gpu+scalar isGPU=1 (want=1) v=6
int8 scalar+gpu isGPU=1 (want=1) v=6
int8 gpu*scalar isGPU=1 v=8
int8 gpu/scalar isGPU=1 v=2
uint8 gpu+scalar isGPU=1 (want=1) v=6
uint8 scalar+gpu isGPU=1 (want=1) v=6
uint8 gpu*scalar isGPU=1 v=8
uint8 gpu/scalar isGPU=1 v=2
int16 gpu+scalar isGPU=1 (want=1) v=6
int16 scalar+gpu isGPU=1 (want=1) v=6
int16 gpu*scalar isGPU=1 v=8
int16 gpu/scalar isGPU=1 v=2
uint16 gpu+scalar isGPU=1 (want=1) v=6
uint16 scalar+gpu isGPU=1 (want=1) v=6
uint16 gpu*scalar isGPU=1 v=8
uint16 gpu/scalar isGPU=1 v=2
int32 gpu+scalar isGPU=1 (want=1) v=6
int32 scalar+gpu isGPU=1 (want=1) v=6
int32 gpu*scalar isGPU=1 v=8
int32 gpu/scalar isGPU=1 v=2
uint32 gpu+scalar isGPU=1 (want=1) v=6
uint32 scalar+gpu isGPU=1 (want=1) v=6
uint32 gpu*scalar isGPU=1 v=8
uint32 gpu/scalar isGPU=1 v=2
int64 gpu+scalar isGPU=1 (want=1) v=6
int64 scalar+gpu isGPU=1 (want=1) v=6
int64 gpu*scalar isGPU=1 v=8
int64 gpu/scalar isGPU=1 v=2
uint64 gpu+scalar isGPU=1 (want=1) v=6
uint64 scalar+gpu isGPU=1 (want=1) v=6
uint64 gpu*scalar isGPU=1 v=8
uint64 gpu/scalar isGPU=1 v=2
cross-device throws: Device mismatch, both NDArray MUST be in the same device.
cross-device throws: Device mismatch, both NDArray MUST be in the same device.
