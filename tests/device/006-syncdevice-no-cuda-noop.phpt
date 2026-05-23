--TEST--
NumPower::syncDevice() is a silent no-op when CUDA is not compiled in
--SKIPIF--
<?php
try {
    (new NDArray([1.0]))->gpu();
    die('skip CUDA is compiled in');
} catch (\Error $e) {
    /* Good — CUDA absent. */
}
?>
--FILE--
<?php
/* The stub documents that syncDevice is a "No-op when CUDA is not
   available" so portable scripts can invoke it unconditionally. */
ob_start();
NumPower::syncDevice();
$out = ob_get_clean();
echo $out === '' ? "ok\n" : "FAIL: produced output $out\n";
?>
--EXPECT--
ok
