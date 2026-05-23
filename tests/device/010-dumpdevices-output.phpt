--TEST--
NumPower::dumpDevices() emits the expected text
--FILE--
<?php
/* On a CUDA build, the dump lists each device with its name and limits.
   On a CPU-only build, it prints a single notice. Both code paths route
   through php_printf so the call must produce *some* output. */
ob_start();
NumPower::dumpDevices();
$out = ob_get_clean();

$cuda_compiled = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $cuda_compiled = false; }

if ($cuda_compiled) {
    $ok = strpos($out, 'Number of CUDA devices') !== false
        && strpos($out, 'Compute capability') !== false
        && strpos($out, 'Total global memory') !== false;
    echo $ok ? "ok\n" : "FAIL: $out\n";
} else {
    $ok = strpos($out, 'No GPU devices available') !== false
        && strpos($out, 'CUDA not enabled')      !== false;
    echo $ok ? "ok\n" : "FAIL: $out\n";
}
?>
--EXPECT--
ok
