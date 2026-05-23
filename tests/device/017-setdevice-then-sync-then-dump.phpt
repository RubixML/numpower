--TEST--
setDevice → GPU work → syncDevice → dumpDevices chain works end-to-end
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Realistic interactive flow: pick a device, queue work, sync, then
   list devices. All three calls must coexist cleanly — sync must not
   leave the runtime in a state that confuses subsequent
   cudaGetDeviceProperties calls, and dumpDevices must not perturb the
   selected device. */
NumPower::setDevice(0);

$a = (new NDArray([1.0, 2.0, 3.0, 4.0], 'float32'))->gpu();
$b = (new NDArray([10.0, 20.0, 30.0, 40.0], 'float32'))->gpu();
$c = NumPower::add($a, $b);

NumPower::syncDevice();

ob_start();
NumPower::dumpDevices();
$dump = ob_get_clean();

$got = $c->cpu()->toArray();
$want = [11.0, 22.0, 33.0, 44.0];

$arith_ok = true;
for ($i = 0; $i < 4; $i++) {
    if (abs($got[$i] - $want[$i]) > 1e-5) $arith_ok = false;
}
$dump_ok = strpos($dump, 'Number of CUDA devices') !== false;

echo ($arith_ok && $dump_ok) ? "ok\n" : "FAIL arith=" . ($arith_ok ? 1 : 0)
    . " dump=" . ($dump_ok ? 1 : 0) . "\n";
?>
--EXPECT--
ok
