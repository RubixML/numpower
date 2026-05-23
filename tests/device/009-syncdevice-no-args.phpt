--TEST--
NumPower::syncDevice() rejects extra arguments
--FILE--
<?php
/* The C wrapper added an explicit ZEND_PARSE_PARAMETERS_START(0, 0).
   Passing any argument must surface as an ArgumentCountError. */
try {
    NumPower::syncDevice(0);
    echo "FAIL: no throw\n";
} catch (\ArgumentCountError $e) {
    echo "ok\n";
} catch (\Error $e) {
    /* Older PHPs may throw plain \Error instead of ArgumentCountError. */
    echo "ok\n";
}
?>
--EXPECT--
ok
