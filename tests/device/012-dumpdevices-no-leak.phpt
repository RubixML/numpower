--TEST--
NumPower::dumpDevices() does not leak across many invocations
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* The function prints metadata only — no PHP allocations, no VRAM
   allocations. Use ob_start to swallow the noise so the test output
   stays clean while still exercising the code path. */
for ($i = 0; $i < 50; $i++) {
    ob_start();
    NumPower::dumpDevices();
    ob_end_clean();
}
echo "ok\n";
?>
--EXPECT--
ok
