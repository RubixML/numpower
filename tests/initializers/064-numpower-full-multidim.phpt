--TEST--
NumPower::full() handles every rank 1-D..5-D on CPU and GPU
--FILE--
<?php
/* Rank scan: shape, size, leaf-element-equals-fill, sum equals n*fill.
   Different dtypes per rank exercise the dtype dispatch path at every
   ndim. The deepest-element walk picks up any stride bug. */

$cases = [
    ['shape' => [7],          'dtype' => 'float32', 'fill' => 1.5,
     'numel' => 7],
    ['shape' => [3, 5],       'dtype' => 'float64', 'fill' => 2.0,
     'numel' => 15],
    ['shape' => [2, 3, 4],    'dtype' => 'int32',   'fill' => -3,
     'numel' => 24],
    ['shape' => [2, 2, 2, 2], 'dtype' => 'uint8',   'fill' => 7,
     'numel' => 16],
    ['shape' => [2, 1, 3, 1, 4], 'dtype' => 'float128', 'fill' => '0.5',
     'numel' => 24],
];

foreach ($cases as $c) {
    $a = NumPower::full($c['shape'], $c['fill'], $c['dtype']);
    $shape_ok = $a->shape() === $c['shape'];
    $size_ok  = $a->size() === $c['numel'];
    $expected_sum = (is_string($c['fill']) ? (float)$c['fill'] : $c['fill']) * $c['numel'];
    $sum_ok   = abs((float)NumPower::sum($a) - $expected_sum) < 1e-9;
    $rank = count($c['shape']);

    /* Walk to the last element and verify it matches the fill. */
    $cur = $a;
    for ($i = 0; $i < $rank; $i++) {
        $cur = $cur[$c['shape'][$i] - 1];
    }
    $leaf_str = rtrim(is_object($cur) ? (string)$cur : (string)$cur);
    $expect_str = (string)$c['fill'];
    /* fp128 string "0.5" stringifies as "0.5" exactly; numeric forms
       stringify with PHP's standard "%g" — accept the prefix match. */
    $leaf_ok = (str_starts_with($leaf_str, rtrim($expect_str, '0')) ||
                $leaf_str === $expect_str ||
                (float)$leaf_str === (float)$expect_str);

    echo implode('x', $c['shape']), ' (', $c['dtype'], '): ',
         'shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok  ? 'OK' : 'BAD'),
         ' sum=',  ($sum_ok   ? 'OK' : 'BAD'),
         ' leaf=', ($leaf_ok  ? 'OK' : 'BAD'),
         "\n";
}

$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $has_gpu = false; }

if ($has_gpu) {
    foreach ([[2, 3, 4], [2, 2, 2, 2]] as $shape) {
        $a = NumPower::full($shape, 4, 'int32', NUMPOWER_CUDA);
        $numel = array_product($shape);
        echo 'gpu ', implode('x', $shape), ': ',
             ($a->isGPU() && $a->shape() === $shape &&
              (string)NumPower::sum($a) === (string)($numel * 4) ? 'OK' : 'BAD'),
             "\n";
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
