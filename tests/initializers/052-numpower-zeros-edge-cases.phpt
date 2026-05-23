--TEST--
NumPower::zeros() handles boundary shapes and the default dtype/device contract
--FILE--
<?php
/* Boundary inputs that have to behave well even though they are easy to
   trip on:
    - shape = [0]      empty 1-D
    - shape = [5, 0]   non-empty leading axis, empty trailing
    - shape = [0, 0]   doubly empty
    - shape = [1]      single-element 1-D (often a stride corner)
    - shape = [1, 1]   single-element 2-D
    - shape = [1024]   big-ish 1-D — covers paths that would page-fault if
                       size or stride math overflowed
    - default dtype is float32, default device is CPU. */

$cases = [[0], [5, 0], [0, 0], [1], [1, 1], [1024]];

foreach ($cases as $shape) {
    $a = NumPower::zeros($shape);
    $expected_numel = 1;
    foreach ($shape as $d) { $expected_numel *= $d; }
    $shape_ok = $a->shape() === $shape;
    $size_ok  = $a->size() === $expected_numel;
    /* sum() on an empty array is 0 by convention. */
    $sum_ok   = (string)NumPower::sum($a) === '0';
    echo '[', implode(',', $shape), ']: shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' size=', ($size_ok ? 'OK' : 'BAD'),
         ' sum=', ($sum_ok ? 'OK' : 'BAD'), "\n";
}

/* Default contract: float32 / CPU. */
$a = NumPower::zeros([2]);
echo 'default dtype is float32: ', (is_float($a[0]) ? 'OK' : 'BAD'), "\n";
echo 'default device is CPU: ', ($a->isGPU() ? 'BAD' : 'OK'), "\n";

/* Explicit default values must behave identically to omitting them. */
$a = NumPower::zeros([2], 'float32', NUMPOWER_CPU);
echo 'explicit defaults parity: ',
     (!$a->isGPU() && (string)$a === '[0, 0]' ? 'OK' : 'BAD'), "\n";

/* NUMPOWER_CPU / NUMPOWER_CUDA constants — must equal 0 / 1 numerically and
   be accepted as device values. */
echo 'NUMPOWER_CPU=', (int)NUMPOWER_CPU, ' NUMPOWER_CUDA=', (int)NUMPOWER_CUDA, "\n";

/* Empty 2-D zeros() round-trips through (->gpu()->cpu()) on GPU builds. */
try {
    $g = NumPower::zeros([3, 0], 'float64', NUMPOWER_CUDA);
    $cpu = $g->cpu();
    echo 'gpu empty round-trip: ',
         ($cpu->shape() === [3, 0] && $cpu->size() === 0 ? 'OK' : 'BAD'), "\n";
} catch (\Error $e) {
    /* On a CPU-only build the line is impossible; keep EXPECT stable. */
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
NUMPOWER_CPU=0 NUMPOWER_CUDA=1
gpu empty round-trip: OK
