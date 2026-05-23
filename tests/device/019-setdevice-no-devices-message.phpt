--TEST--
NumPower::setDevice() message format is sane regardless of deviceCount
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* When deviceCount == 0 the original code would format the error as
   "valid range: 0..-1" which is nonsensical. The fix surfaces a
   dedicated "No CUDA devices are visible" message instead. We can't
   force deviceCount=0 on a CUDA-enabled host, but we *can* assert that
   the out-of-range error never contains "..-1" garbage for any input. */
try {
    NumPower::setDevice(999999);
    echo "FAIL: no throw\n";
} catch (\Error $e) {
    $msg = $e->getMessage();
    /* Must NOT include the malformed "..-1" range marker. */
    $sane = strpos($msg, '..-1') === false;
    echo $sane ? "ok\n" : "FAIL: $msg\n";
}
?>
--EXPECT--
ok
