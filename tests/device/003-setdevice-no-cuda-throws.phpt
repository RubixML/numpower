--TEST--
NumPower::setDevice() throws when CUDA is not compiled in
--SKIPIF--
<?php
/* Inverse skip: only run when CUDA is *not* compiled. */
try {
    (new NDArray([1.0]))->gpu();
    die('skip CUDA is compiled in');
} catch (\Error $e) {
    /* Good — CUDA absent, run the test. */
}
?>
--FILE--
<?php
/* The stub documents `@throws \Error When CUDA is not compiled in`.
   Before the fix the call was a silent no-op, violating that contract. */
try {
    NumPower::setDevice(0);
    echo "FAIL: no throw\n";
} catch (\Error $e) {
    $msg = $e->getMessage();
    $ok = strpos($msg, 'CUDA not enabled') !== false
        || strpos($msg, 'No GPU device available') !== false;
    echo $ok ? "ok\n" : "FAIL: $msg\n";
}
?>
--EXPECT--
ok
