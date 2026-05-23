--TEST--
syncDevice() in a portable CPU+GPU script never throws on idle device
--FILE--
<?php
/* Realistic usage pattern: build CPU data, optionally promote to GPU,
   sync before reading. syncDevice() must be safe to sprinkle into
   scripts regardless of whether a GPU is present. */
$cuda_compiled = true;
try {
    (new NDArray([1.0]))->gpu();
} catch (\Error $e) {
    $cuda_compiled = false;
}

$a = new NDArray([1.0, 2.0, 3.0, 4.0], 'float32');
if ($cuda_compiled) {
    $a = $a->gpu();
}

NumPower::syncDevice();  /* must be a no-op on CPU build, harmless on GPU */

if ($cuda_compiled) {
    $a = $a->cpu();
}
$got = $a->toArray();
$want = [1.0, 2.0, 3.0, 4.0];
$ok = true;
for ($i = 0; $i < 4; $i++) {
    if (abs($got[$i] - $want[$i]) > 1e-6) $ok = false;
}
echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
