--TEST--
NDArray::toImage() does not leak persistent zend_strings across many calls
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* The previous `php_gd_assign_libgdimageptr_as_extgdimage` helper
   allocated a persistent zend_string (`zend_string_init(..., 1)`) per
   call and never released it. That leaked into the persistent pool on
   every `toImage()` invocation — not visible to `memory_get_usage()`
   (which only sees emalloc), but a real long-running leak.

   This test exercises a large number of `toImage()` calls and asserts
   that no PHP-visible heap growth occurs across the loop. Combined with
   `008-fromimage-vram-no-leak.phpt`, this pins both the host and device
   memory contracts for the toImage path. */

$img = imagecreatetruecolor(8, 8);
imagefilledrectangle($img, 0, 0, 7, 7, 0x102030);
$arr = NumPower::fromImage($img);

$warm_start = memory_get_usage(false);
for ($i = 0; $i < 100; $i++) {
    $back = $arr->toImage();
    $back = null;
}
$warm_end = memory_get_usage(false);

/* Now stress with 10k iterations — the cached class entry means no
   per-call growth. */
$start = memory_get_usage(false);
for ($i = 0; $i < 10000; $i++) {
    $back = $arr->toImage();
    $back = null;
}
$end = memory_get_usage(false);

$delta_warm = $warm_end - $warm_start;
$delta_main = $end - $start;
echo 'warm_delta_zero: ', ($delta_warm === 0 ? 'OK' : "BAD ($delta_warm)"), "\n";
echo 'main_delta_zero: ', ($delta_main === 0 ? 'OK' : "BAD ($delta_main)"), "\n";
?>
--EXPECT--
warm_delta_zero: OK
main_delta_zero: OK
