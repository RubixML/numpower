--TEST--
NumPower::syncDevice() does not leak across many invocations
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* The C wrapper allocates nothing and the underlying cudaDeviceSynchronize
   does not own host memory, so a tight loop must stay flat in both PHP
   memory and VRAM. The VRAM leak hook at RSHUTDOWN prints
   "VRAM MEMORY LEAK" otherwise. */
for ($i = 0; $i < 1000; $i++) {
    NumPower::syncDevice();
}
echo "ok\n";
?>
--EXPECT--
ok
