--TEST--
NumPower::dumpDevices() output has balanced borders on a CUDA build
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The CUDA branch prints an outer "===" border, then per-device "---"
   sections, then a closing "===" border. A previous version returned
   early from the per-device loop on cudaGetDeviceProperties errors and
   skipped the closing border. Assert: at least one of each border
   appears, and the dump ends with the closing "===" line. */
ob_start();
NumPower::dumpDevices();
$out = ob_get_clean();

$eq_count = substr_count($out, "==============================================================================");
$dash_count = substr_count($out, "---------------------------------------------------------------------------");

/* Open border + close border = 2 minimum. Each device prints two dash
   borders (open + close), so at least 2 (for one device). */
$borders_ok = $eq_count >= 2 && $dash_count >= 2;

/* Output must end with the closing equals border followed by newline. */
$trimmed = rtrim($out, "\n");
$ends_with_close = substr($trimmed, -78)
    === "==============================================================================";

echo ($borders_ok && $ends_with_close) ? "ok\n"
   : "FAIL eq=$eq_count dash=$dash_count ends_close="
     . ($ends_with_close ? 1 : 0) . "\n";
?>
--EXPECT--
ok
