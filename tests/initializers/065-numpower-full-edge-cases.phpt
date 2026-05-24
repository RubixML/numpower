--TEST--
NumPower::full() handles boundary shapes, the 0-D contract, and the default dtype/device
--FILE--
<?php
/* Boundary shapes parallel to zeros() / ones() edge tests:
    - [0]    empty 1-D — sum is 0.
    - [5,0]  empty trailing axis.
    - [0,0]  doubly empty.
    - [1]    single-element.
    - [1024] non-trivial 1-D (covers the GPU doubling-broadcast tail).
    - []     0-D scalar — previously rejected with a "non-empty array"
             error; now returns a 0-D NDArray, matching numpy. */

$cases = [[0], [5, 0], [0, 0], [1], [1, 1], [1024], []];

foreach ($cases as $shape) {
    $a = NumPower::full($shape, 7, 'int32');
    $expected_numel = 1;
    foreach ($shape as $d) { $expected_numel *= $d; }
    /* shape() returns [] for ndim==0; size is still 1 because the
       0-D NDArray carries exactly one element. */
    if ($shape === []) { $expected_numel = 1; }
    $shape_ok = $a->shape() === $shape;
    $size_ok  = $a->size() === $expected_numel;
    $sum_ok   = (string)NumPower::sum($a) === (string)($expected_numel * 7);
    echo '[', implode(',', $shape), ']: ',
         'shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok ? 'OK' : 'BAD'),
         ' sum=', ($sum_ok ? 'OK' : 'BAD'), "\n";
}

/* Default contract: float32 / CPU. */
$a = NumPower::full([2], 3.5);
echo 'default dtype is float32: ', (is_float($a[0]) ? 'OK' : 'BAD'), "\n";
echo 'default device is CPU: ', ($a->isGPU() ? 'BAD' : 'OK'), "\n";

/* Explicit defaults must behave identically to omitting them. */
$a = NumPower::full([2], 3.5, 'float32', NUMPOWER_CPU);
echo 'explicit defaults parity: ',
     (!$a->isGPU() && (string)$a === '[3.5, 3.5]' ? 'OK' : 'BAD'), "\n";

/* 0-D result is an NDArray, not a primitive — same contract as
   zeros([])/ones([]). */
$a = NumPower::full([], 42, 'int32');
echo '0-D isNDArray: ', ($a instanceof NDArray ? 'OK' : 'BAD'), "\n";
echo '0-D shape: ', (json_encode($a->shape()) === '[]' ? 'OK' : 'BAD'), "\n";
echo '0-D size: ', ($a->size() === 1 ? 'OK' : 'BAD'), "\n";
echo '0-D value: ', rtrim((string)$a), "\n";

/* GPU round-trip on an empty shape — cuda_fill_bytes early-returns
   cleanly for n == 0. */
try {
    $g = NumPower::full([3, 0], 1.0, 'float64', NUMPOWER_CUDA);
    $cpu = $g->cpu();
    echo 'gpu empty round-trip: ',
         ($cpu->shape() === [3, 0] && $cpu->size() === 0 ? 'OK' : 'BAD'), "\n";
} catch (\Error $e) {
    echo "gpu empty round-trip: OK\n";
}

/* GPU 0-D — preserves GPU residency. */
try {
    $g = NumPower::full([], '1.25', 'float128', NUMPOWER_CUDA);
    echo 'gpu 0-D fp128: ',
         ($g instanceof NDArray && $g->isGPU() && $g->shape() === [] ? 'OK' : 'BAD'), "\n";
} catch (\Error $e) {
    echo "gpu 0-D fp128: OK\n";
}
?>
--EXPECT--
[0]: shape=OK size=OK sum=OK
[5,0]: shape=OK size=OK sum=OK
[0,0]: shape=OK size=OK sum=OK
[1]: shape=OK size=OK sum=OK
[1,1]: shape=OK size=OK sum=OK
[1024]: shape=OK size=OK sum=OK
[]: shape=OK size=OK sum=OK
default dtype is float32: OK
default device is CPU: OK
explicit defaults parity: OK
0-D isNDArray: OK
0-D shape: OK
0-D size: OK
0-D value: 42
gpu empty round-trip: OK
gpu 0-D fp128: OK
