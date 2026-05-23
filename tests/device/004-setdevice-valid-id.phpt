--TEST--
NumPower::setDevice(0) succeeds on any CUDA-capable build
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Device 0 is guaranteed to exist on every CUDA box that ran the SKIPIF
   probe. Set, then do a small GPU computation to confirm the device is
   still usable after the explicit selection. */
NumPower::setDevice(0);

$a = (new NDArray([1.0, 2.0, 3.0, 4.0], 'float32'))->gpu();
$b = (new NDArray([10.0, 20.0, 30.0, 40.0], 'float32'))->gpu();
$c = NumPower::add($a, $b)->cpu();

$want = [11.0, 22.0, 33.0, 44.0];
$got  = $c->toArray();
for ($i = 0; $i < 4; $i++) {
    if (abs($want[$i] - $got[$i]) > 1e-5) {
        echo "FAIL: idx=$i want={$want[$i]} got={$got[$i]}\n";
        exit;
    }
}
echo "ok\n";
?>
--EXPECT--
ok
