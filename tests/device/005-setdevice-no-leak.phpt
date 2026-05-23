--TEST--
NumPower::setDevice() does not allocate or leak across many invocations
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* setDevice() touches only the CUDA runtime state — no PHP allocations,
   no VRAM allocations. The VRAM leak hook at RSHUTDOWN prints
   "VRAM MEMORY LEAK" if it sees an unfreed device buffer; the suite
   passes only if it stays silent. */

for ($i = 0; $i < 200; $i++) {
    try {
        NumPower::setDevice(0);
    } catch (\Error $e) {
        /* Expected on no-CUDA builds, irrelevant to the leak check. */
    }
    /* Also exercise the throw paths, which must not leak either. */
    try { NumPower::setDevice(-1); } catch (\Error $e) {}
    try { NumPower::setDevice(1000000); } catch (\Error $e) {}
}
echo "ok\n";
?>
--EXPECT--
ok
