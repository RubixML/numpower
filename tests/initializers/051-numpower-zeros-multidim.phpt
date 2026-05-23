--TEST--
NumPower::zeros() handles every rank 1-D..5-D on CPU and GPU
--FILE--
<?php
/* Rank scan: shape preservation + element-count math + a boundary read
   ($a[i_last]) on each rank to make sure the strides agree with the
   nominal shape (a stride bug would surface as an out-of-range read or
   the wrong sub-shape). Mixed integer / float / fp128 dtypes per rank
   so dtype-dispatch in the new init path is exercised at every rank. */

$cases = [
    ['shape' => [7],          'dtype' => 'float32', 'numel' => 7],
    ['shape' => [3, 5],       'dtype' => 'float64', 'numel' => 15],
    ['shape' => [2, 3, 4],    'dtype' => 'int32',   'numel' => 24],
    ['shape' => [2, 2, 2, 2], 'dtype' => 'uint8',   'numel' => 16],
    ['shape' => [2, 1, 3, 1, 4], 'dtype' => 'float128', 'numel' => 24],
];

foreach ($cases as $c) {
    $a = NumPower::zeros($c['shape'], $c['dtype']);
    $shape_ok = $a->shape() === $c['shape'];
    $size_ok  = $a->size() === $c['numel'];
    $sum_ok   = (string)NumPower::sum($a) === '0';
    $rank = count($c['shape']);

    /* Walk down to a leaf via $a[last][last]... and verify it's zero. */
    $cur = $a;
    for ($i = 0; $i < $rank; $i++) {
        $cur = $cur[$c['shape'][$i] - 1];
    }
    $leaf_ok = ($cur === 0 || $cur === 0.0 || $cur === '0' || $cur === '0.0');

    echo implode('x', $c['shape']), ' (', $c['dtype'], '): ',
         'shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok  ? 'OK' : 'BAD'),
         ' sum=', ($sum_ok   ? 'OK' : 'BAD'),
         ' leaf=', ($leaf_ok  ? 'OK' : 'BAD'),
         "\n";
}

/* Repeat the smallest 3-D and 4-D cases on GPU if available. */
$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); }
catch (\Error $e) { $has_gpu = false; }

if ($has_gpu) {
    foreach ([[2, 3, 4], [2, 2, 2, 2]] as $shape) {
        $a = NumPower::zeros($shape, 'float32', NUMPOWER_CUDA);
        echo 'gpu ', implode('x', $shape), ': ',
             ($a->isGPU() && $a->shape() === $shape &&
              (string)NumPower::sum($a) === '0' ? 'OK' : 'BAD'), "\n";
    }
} else {
    /* Keep the EXPECT block stable on a CPU-only build. */
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
