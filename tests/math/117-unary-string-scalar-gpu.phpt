--TEST--
String-scalar intake works alongside GPU NDArrays (clip with GPU input + string bounds; .gpu() chains on the result)
--SKIPIF--
<?php
try {
    $a = NumPower::array([1.0])->gpu();
    if (!$a->isGPU()) die("skip GPU not available");
} catch (Throwable $t) {
    die("skip GPU not available: " . $t->getMessage());
}
?>
--FILE--
<?php
/* The bare-string-scalar intake builds a CPU 0-D NDArray internally,
   so the result lives on CPU. Verify two compositions:

   1) GPU NDArray + string `min`/`max` bounds: clip's bound parser
      accepts strings on every device — the array stays on GPU and the
      bounds parse via NDArray_TypedUnaryOp's per-dtype path.

   2) String-scalar unary result can be moved to GPU via `.gpu()` —
      sanity check that the CPU 0-D NDArray returned by the string
      intake is a valid intake to GPU staging. */

/* (1) GPU array + string bounds. */
$arr = NumPower::array([-2.0, 0.5, 3.0], 'float64')->gpu();
$out = NumPower::clip($arr, '-1.0', '2.0');
if (!$out->isGPU()) echo "FAIL clip-gpu lost device\n";
else                echo "OK clip-gpu(string bounds) stayed on GPU\n";
echo "clip-gpu(string bounds) values: ", json_encode($out->cpu()->toArray()), "\n";

/* fp128 GPU array + fp128 string bounds. */
$f = NumPower::array(['-1e30', '0', '1e30'], 'float128')->gpu();
$cl = NumPower::clip($f, '0', '1e29');
if (!$cl->isGPU()) echo "FAIL fp128 clip-gpu lost device\n";
else               echo "OK fp128 clip-gpu(string bounds) stayed on GPU\n";

/* (2) String-scalar intake produces a CPU NDArray result wrapped in a
   dtype-aware scalar (string for fp128 here). Build a fresh fp128
   NDArray from that string and move it to GPU. */
$s = NumPower::exp('1.0');                 /* string ≈ "2.71828..." */
if (!is_string($s)) {
    echo "FAIL exp('1.0') expected fp128 string scalar, got ",
         gettype($s), "\n";
} else {
    $n  = new NDArray([$s], 'float128');
    $ng = $n->gpu();
    echo "OK exp('1.0') fp128 string → NDArray fp128 → GPU isGPU=",
         (int)$ng->isGPU(), "\n";
}

echo "DONE\n";
?>
--EXPECTF--
OK clip-gpu(string bounds) stayed on GPU
clip-gpu(string bounds) values: [-1,0.5,2]
OK fp128 clip-gpu(string bounds) stayed on GPU
OK exp('1.0') fp128 string → NDArray fp128 → GPU isGPU=1
DONE
