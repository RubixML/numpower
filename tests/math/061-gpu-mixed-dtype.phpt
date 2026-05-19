--TEST--
GPU mixed-dtype arithmetic uses GPU AsType (no CPU round-trip for any pair of native dtypes)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* When operands have different dtypes, the dispatcher promotes and casts on
   GPU via the AsType kernel rather than pulling everything to CPU. The
   common pairs below all stay on GPU. */

$pairs = [
    ['int8',    'int32'],
    ['int32',   'float64'],
    ['uint8',   'float32'],
    ['float16', 'float64'],
    ['int64',   'float32'],
];

foreach ($pairs as [$ta, $tb]) {
    $a = NumPower::array([4, 4], $ta)->gpu();
    $b = NumPower::array([2, 2], $tb)->gpu();
    $r = $a + $b;
    echo "$ta + $tb: isGPU=", $r->isGPU(), " v=", (string)$r->cpu()[0], "\n";
}

/* GPU int32 / 2 must promote to float on GPU and stay on GPU (PyTorch
   true-division semantics). */
$a = NumPower::array([5, 5, 5, 5], 'int32')->gpu();
$r = $a / 2;
echo "int32 GPU / 2: isGPU=", $r->isGPU(), " v=", (string)$r->cpu()[0], "\n";

/* int8 GPU * scalar fp32 — scalar promotes via weak-scalar to int8, result
   stays int8 on GPU. */
$a = NumPower::array([5, 5, 5, 5], 'int8')->gpu();
$r = $a + 3;
echo "int8 GPU + 3: isGPU=", $r->isGPU(), " v=", (string)$r->cpu()[0], "\n";
?>
--EXPECT--
int8 + int32: isGPU=1 v=6
int32 + float64: isGPU=1 v=6
uint8 + float32: isGPU=1 v=6
float16 + float64: isGPU=1 v=6
int64 + float32: isGPU=1 v=6
int32 GPU / 2: isGPU=1 v=2.5
int8 GPU + 3: isGPU=1 v=8
