--TEST--
NumPower::setDevice(PHP_INT_MAX) throws cleanly (no integer overflow on (int) cast)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The C side parses deviceId as zend_long (64-bit) and casts to int
   (32-bit) only after the range check passes. With PHP_INT_MAX the
   range check trips first, so no narrowing cast is performed; the test
   guards against a future refactor that reorders the cast and silently
   wraps to a small valid id. */
try {
    NumPower::setDevice(PHP_INT_MAX);
    echo "FAIL: no throw\n";
} catch (\Error $e) {
    $msg = $e->getMessage();
    /* Must mention the offending id AND identify it as out of range. */
    $ok = strpos($msg, (string)PHP_INT_MAX) !== false
        && strpos($msg, 'does not exist') !== false;
    echo $ok ? "ok\n" : "FAIL: $msg\n";
}
?>
--EXPECT--
ok
