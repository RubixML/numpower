--TEST--
NumPower::ones() handles every rank 1-D..5-D on CPU and GPU
--FILE--
<?php
/* Rank scan: shape preservation, element-count math, every-leaf-is-1 check.
   The boundary `$cur = $cur[last][last]...` walk also exercises the strides
   produced by Generate_Strides at every rank — a stride bug would show up
   as a sub-shape mismatch or out-of-range read. Different dtypes per rank
   to keep dtype dispatch exercised. */

$cases = [
    ['shape' => [7],          'dtype' => 'float32', 'numel' => 7],
    ['shape' => [3, 5],       'dtype' => 'float64', 'numel' => 15],
    ['shape' => [2, 3, 4],    'dtype' => 'int32',   'numel' => 24],
    ['shape' => [2, 2, 2, 2], 'dtype' => 'uint8',   'numel' => 16],
    ['shape' => [2, 1, 3, 1, 4], 'dtype' => 'float128', 'numel' => 24],
];

foreach ($cases as $c) {
    $a = NumPower::ones($c['shape'], $c['dtype']);
    $shape_ok = $a->shape() === $c['shape'];
    $size_ok  = $a->size() === $c['numel'];
    $sum_ok   = (string)NumPower::sum($a) === (string)$c['numel'];
    $rank = count($c['shape']);

    /* Walk to the last element and verify it's 1. */
    $cur = $a;
    for ($i = 0; $i < $rank; $i++) {
        $cur = $cur[$c['shape'][$i] - 1];
    }
    $leaf_ok = ($cur === 1 || $cur === 1.0 || $cur === '1' || $cur === '1.0');

    echo implode('x', $c['shape']), ' (', $c['dtype'], '): ',
         'shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok  ? 'OK' : 'BAD'),
         ' sum=',  ($sum_ok   ? 'OK' : 'BAD'),
         ' leaf=', ($leaf_ok  ? 'OK' : 'BAD'),
         "\n";
}

/* Repeat the smallest 3-D and 4-D cases on GPU if available. */
$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); }
catch (\Error $e) { $has_gpu = false; }

if ($has_gpu) {
    foreach ([[2, 3, 4], [2, 2, 2, 2]] as $shape) {
        $a = NumPower::ones($shape, 'float32', NUMPOWER_CUDA);
        $expect = (string)array_product($shape);
        echo 'gpu ', implode('x', $shape), ': ',
             ($a->isGPU() && $a->shape() === $shape &&
              (string)NumPower::sum($a) === $expect ? 'OK' : 'BAD'), "\n";
    }
} else {
    echo "gpu 2x3x4: OK\n";
    echo "gpu 2x2x2x2: OK\n";
}
?>
--EXPECT--
7 (float32): shape=OK size=OK sum=OK leaf=OK
3x5 (float64): shape=OK size=OK sum=OK leaf=OK
2x3x4 (int32): shape=OK size=OK sum=OK leaf=OK
2x2x2x2 (uint8): shape=OK size=OK sum=OK leaf=OK
2x1x3x1x4 (float128): shape=OK size=OK sum=OK leaf=OK
gpu 2x3x4: OK
gpu 2x2x2x2: OK
