--TEST--
NumPower::setDevice() rejects ids >= deviceCount
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Any GPU box has < 1_000_000 devices, so id 1_000_000 is reliably out of
   range. The exact message format is "Device N does not exist
   (valid range: 0..M)" — assert both halves. */
try {
    NumPower::setDevice(1000000);
    echo "FAIL: no throw\n";
} catch (\Error $e) {
    $msg = $e->getMessage();
    $ok = strpos($msg, 'does not exist') !== false
        && strpos($msg, 'valid range') !== false;
    echo $ok ? "ok\n" : "FAIL: $msg\n";
}
?>
--EXPECT--
ok
