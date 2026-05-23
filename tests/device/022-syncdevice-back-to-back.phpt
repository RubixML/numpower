--TEST--
NumPower::syncDevice() can be called back-to-back without state corruption
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Calling syncDevice() repeatedly with no work in between must be a
   cheap no-op. The previous (pre-fix) version would swallow CUDA
   return codes, so we also verify that the second sync sees a clean
   state — no sticky error from the first. */
NumPower::syncDevice();
NumPower::syncDevice();
NumPower::syncDevice();
NumPower::syncDevice();

/* Queue a small amount of work, then double-sync. */
$a = (new NDArray([1.0, 2.0, 3.0, 4.0], 'float32'))->gpu();
$b = (new NDArray([0.5, 0.5, 0.5, 0.5], 'float32'))->gpu();
$c = NumPower::add($a, $b);

NumPower::syncDevice();
NumPower::syncDevice();  /* second sync must succeed and not throw */

$got = $c->cpu()->toArray();
echo ($got === [1.5, 2.5, 3.5, 4.5]) ? "ok\n" : "FAIL: " . json_encode($got) . "\n";
?>
--EXPECT--
ok
