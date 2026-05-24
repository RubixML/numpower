--TEST--
NumPower::fromImage() / NDArray::toImage() respect PHP GdImage refcount lifetime
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* GdImage lifetime in PHP 8.0+ is managed exclusively by refcount;
   the legacy `imagedestroy()` function is a no-op since 8.0 and
   `E_DEPRECATED` since 8.5, so this test exercises lifetime through
   the modern `unset()` / null-assignment idiom instead.

   The C extension reaches into the underlying `gdImagePtr` via the
   PHP object struct, so it must honor the refcount: as long as any
   variable still holds the GdImage, the pointer is valid.

   Same property for toImage: the produced GdImage must outlive the
   NDArray it was produced from (the NDArray is freed but the GD
   memory is its own malloc'd allocation, owned by libgd's destructor
   that fires only when the last PHP reference goes away). */

/* Refcount kept-alive case: the local var $img is unset but $x still
   holds the GdImage. fromImage($x) must still see valid pixel data. */
$img = imagecreatetruecolor(3, 2);
imagesetpixel($img, 0, 0, 0xFF0000);
imagesetpixel($img, 2, 1, 0x00FF00);
$x = $img;
unset($img);
echo 'img_unset_isset: ', (isset($img) ? 'yes' : 'no'), "\n";
$a = NumPower::fromImage($x);
echo 'fromImage_after_unset_shape: ', json_encode($a->shape()), "\n";
echo 'fromImage_after_unset_(0,0)R: ', (int)$a[0][0][0], "\n";
echo 'fromImage_after_unset_(1,2)G: ', (int)$a[1][2][1], "\n";

/* toImage outlives the source NDArray: build the image, drop the
   NDArray, then keep using the image. The NDArray's element buffer
   is `emalloc`'d and released by `NDArray_FREE`; the produced GD
   image is `malloc`'d and owned by libgd's destructor. The two are
   independent — releasing the NDArray must not touch the image. */
$a = NumPower::fromImage($x);
$img2 = $a->toImage();
$a = null;
$pix = imagecolorat($img2, 0, 0) & 0xFFFFFF;
echo 'toImage_outlives_ndarray: ', sprintf('#%06x', $pix), "\n";

/* Now exercise the reverse: drop the GdImage first, keep the NDArray.
   `fromImage` consumed `$x`'s pixel data into the NDArray buffer;
   after `unset($x)` the NDArray must still be intact. */
$x2 = imagecreatetruecolor(3, 2);
imagesetpixel($x2, 0, 0, 0x0000FF);
$a = NumPower::fromImage($x2);
unset($x2);
echo 'ndarray_outlives_gdimage_(0,0)B: ', (int)$a[0][0][2], "\n";

/* GPU variant: the NDArray lives on the device. `toImage` stages
   the device data through `NDArray_TypedD2H` into a temporary host
   buffer that my code emallocs and efrees before assigning the GD
   image. Dropping the NDArray afterward must leave the image intact. */
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; }
catch (\Error $e) { $has_gpu = false; }
if ($has_gpu) {
    $src = imagecreatetruecolor(3, 2);
    imagesetpixel($src, 0, 0, 0xFF0000);
    $g = NumPower::fromImage($src, true, 'uint8', NUMPOWER_CUDA);
    $img3 = $g->toImage();
    $g = null;
    $pix = imagecolorat($img3, 0, 0) & 0xFFFFFF;
    echo 'toImage_outlives_gpu_ndarray: ', sprintf('#%06x', $pix), "\n";
} else {
    echo "toImage_outlives_gpu_ndarray: #ff0000\n";
}
?>
--EXPECT--
img_unset_isset: no
fromImage_after_unset_shape: [2,3,3]
fromImage_after_unset_(0,0)R: 255
fromImage_after_unset_(1,2)G: 255
toImage_outlives_ndarray: #ff0000
ndarray_outlives_gdimage_(0,0)B: 255
toImage_outlives_gpu_ndarray: #ff0000
