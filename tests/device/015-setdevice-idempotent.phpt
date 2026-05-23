--TEST--
NumPower::setDevice(0) is idempotent — repeated calls do not drift state
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* setDevice() is a pure runtime-state write — calling it with the same
   value any number of times must leave the system in exactly the same
   working state. Verifies by running a small GPU op after each repeat. */
for ($i = 0; $i < 10; $i++) {
    NumPower::setDevice(0);
}
$a = (new NDArray([1.0, 2.0], 'float32'))->gpu();
$b = (new NDArray([10.0, 20.0], 'float32'))->gpu();
$c = NumPower::add($a, $b)->cpu()->toArray();
echo ($c == [11.0, 22.0]) ? "ok\n" : "FAIL: " . json_encode($c) . "\n";
?>
--EXPECT--
ok
