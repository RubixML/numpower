--TEST--
NumPower::fromImage() / NDArray::toImage() handle PHP-8 GdImage refcounting correctly
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* In PHP 8.x, `imagedestroy()` releases the *local* reference but
   leaves the GdImage alive if any other ref holds it. The C extension
   reaches into the underlying `gdImagePtr` via the PHP object struct,
   so this lifetime must be honored. Also covers: passing a GdImage that
   was held in a local var → the var is unset → the underlying image is
   destroyed only when the last reference goes away.

   Same for toImage: the produced GdImage must outlive the NDArray it
   was produced from (the NDArray is freed but the GD memory was its
   own allocation via malloc, owned by libgd, freed by PHP's GdImage
   destructor). */

/* Refcount: the local var $img is destroyed but $x still holds the
   GdImage. fromImage($x) should still work. */
$img = imagecreatetruecolor(3, 2);
imagesetpixel($img, 0, 0, 0xFF0000);
imagesetpixel($img, 2, 1, 0x00FF00);
$x = $img;
imagedestroy($img); // PHP 8: no-op; var $img still points to the object
echo '$img_type_after_destroy: ', gettype($img), "\n";
$a = NumPower::fromImage($x);
echo 'fromImage_after_destroy: ', json_encode($a->shape()), "\n";
echo '(0,0): ', (int)$a[0][0][0], " (expect 255)\n";

/* toImage outlives the source NDArray: build the image, drop the
   NDArray, then keep using the image. The image's memory is allocated
   by my code via malloc and owned by libgd's destructor — releasing
   the NDArray's data buffer (emalloc) must not touch the image. */
$a = NumPower::fromImage($x);
$img2 = $a->toImage();
$a = null; // NDArray freed
$pix = imagecolorat($img2, 0, 0) & 0xFFFFFF;
echo 'toImage_outlives_ndarray: ', sprintf('#%06x (expect #ff0000)', $pix), "\n";

/* Same on GPU: the NDArray is on the device, so its host pointer is
   NULL; staging copies the device data to a temp host buffer which my
   code emallocs and efrees. The produced GD image must not reference
   any of that staging. */
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; }
catch (\Error $e) { $has_gpu = false; }
if ($has_gpu) {
    $g = NumPower::fromImage($x, true, 'uint8', NUMPOWER_CUDA);
    $img3 = $g->toImage();
    $g = null;
    $pix = imagecolorat($img3, 0, 0) & 0xFFFFFF;
    echo 'toImage_outlives_gpu_ndarray: ', sprintf('#%06x (expect #ff0000)', $pix), "\n";
} else {
    echo "toImage_outlives_gpu_ndarray: #ff0000 (expect #ff0000)\n";
}
?>
--EXPECT--
$img_type_after_destroy: object
fromImage_after_destroy: [2,3,3]
(0,0): 255 (expect 255)
toImage_outlives_ndarray: #ff0000 (expect #ff0000)
toImage_outlives_gpu_ndarray: #ff0000 (expect #ff0000)
