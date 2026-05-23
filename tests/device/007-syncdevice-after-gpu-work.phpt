--TEST--
NumPower::syncDevice() works as a barrier after queued GPU work
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Queue a non-trivial chain of GPU work, then call syncDevice() to ensure
   all kernels have finished. The interleaving of work + sync + read must
   yield numerically correct results regardless of how the runtime chose
   to schedule the underlying kernels. */
$n  = 4096;
$xs = array_fill(0, $n, 1.5);
$ys = array_fill(0, $n, 2.5);

$a = (new NDArray($xs, 'float32'))->gpu();
$b = (new NDArray($ys, 'float32'))->gpu();

for ($i = 0; $i < 16; $i++) {
    $a = NumPower::add($a, $b);
}

NumPower::syncDevice();  /* must not throw, must not leak */

$got = $a->cpu()->toArray();
$want = 1.5 + 16 * 2.5;
foreach ($got as $idx => $v) {
    if (abs($v - $want) > 1e-3) {
        echo "FAIL: idx=$idx want=$want got=$v\n";
        exit;
    }
}
echo "ok\n";
?>
--EXPECT--
ok
