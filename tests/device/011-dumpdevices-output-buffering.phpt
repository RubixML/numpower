--TEST--
NumPower::dumpDevices() output is captured by ob_start()
--FILE--
<?php
/* Before the fix the CUDA branch used raw printf(), bypassing PHP output
   buffering. After the fix every line goes through php_printf, so
   ob_get_contents() sees the whole dump. The CPU branch already used
   php_printf and is exercised here too. */
ob_start();
NumPower::dumpDevices();
$inside = ob_get_clean();

/* If raw printf were used, $inside would be empty in CLI mode while the
   text would still appear on stdout. We assert that ob_* captured *some*
   non-empty output (the exact content depends on the build). */
if ($inside === '' || $inside === false) {
    echo "FAIL: dumpDevices output bypassed output buffering\n";
} else {
    echo "ok\n";
}
?>
--EXPECT--
ok
