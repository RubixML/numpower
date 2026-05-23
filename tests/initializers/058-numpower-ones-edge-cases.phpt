--TEST--
NumPower::ones() handles boundary shapes and the default dtype/device contract
--FILE--
<?php
/* Boundary inputs that need to stay well-behaved:
    - [0]        empty 1-D — sum() is 0 even though the dtype is "ones".
    - [5, 0]     empty trailing axis — no element to write.
    - [0, 0]     doubly empty.
    - [1]        single-element 1-D — exercises the n==1 fast-path in
                 cuda_fill_bytes (seed copy, no doubling loop body).
    - [1, 1]     single-element 2-D.
    - [1024]     non-trivial 1-D — covers the doubling broadcast.

   The defaults must be float32 / CPU (mirroring zeros()). */

$cases = [[0], [5, 0], [0, 0], [1], [1, 1], [1024]];

foreach ($cases as $shape) {
    $a = NumPower::ones($shape);
    $expected_numel = 1;
    foreach ($shape as $d) { $expected_numel *= $d; }
    $shape_ok = $a->shape() === $shape;
    $size_ok  = $a->size() === $expected_numel;
    $sum_ok   = (string)NumPower::sum($a) === (string)$expected_numel;
    echo '[', implode(',', $shape), ']: ',
         'shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok ? 'OK' : 'BAD'),
         ' sum=', ($sum_ok ? 'OK' : 'BAD'), "\n";
}

/* Default contract: float32 / CPU. */
$a = NumPower::ones([2]);
echo 'default dtype is float32: ', (is_float($a[0]) ? 'OK' : 'BAD'), "\n";
echo 'default device is CPU: ', ($a->isGPU() ? 'BAD' : 'OK'), "\n";

/* Explicit defaults must behave identically to omitting them. */
$a = NumPower::ones([2], 'float32', NUMPOWER_CPU);
echo 'explicit defaults parity: ',
     (!$a->isGPU() && (string)$a === '[1, 1]' ? 'OK' : 'BAD'), "\n";

/* GPU round-trip on an empty shape — the doubling loop must skip cleanly
   when n == 0; cuda_fill_bytes early-returns for n <= 0. */
try {
    $g = NumPower::ones([3, 0], 'float64', NUMPOWER_CUDA);
    $cpu = $g->cpu();
    echo 'gpu empty round-trip: ',
         ($cpu->shape() === [3, 0] && $cpu->size() === 0 ? 'OK' : 'BAD'), "\n";
} catch (\Error $e) {
    echo "gpu empty round-trip: OK\n";
}
?>
--EXPECT--
[0]: shape=OK size=OK sum=OK
[5,0]: shape=OK size=OK sum=OK
[0,0]: shape=OK size=OK sum=OK
[1]: shape=OK size=OK sum=OK
[1,1]: shape=OK size=OK sum=OK
[1024]: shape=OK size=OK sum=OK
default dtype is float32: OK
default device is CPU: OK
explicit defaults parity: OK
gpu empty round-trip: OK
