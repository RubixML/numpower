--TEST--
NDArray::toImage() handles non-contiguous GPU NDArrays (transposed/sliced views)
--SKIPIF--
<?php
if (!extension_loaded('gd')) die('skip GD extension not loaded');
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The GPU staging path stages a contiguous `n_elements * elsize` block
   from the device. For a transposed *view* of a contiguous parent the
   data pointer is the parent's start and the view's strides translate
   indices to byte offsets within the staged copy — the test pins that
   layout-permutation pipelines do not corrupt pixels when the input is
   a non-contiguous GPU view. */

$src = imagecreatetruecolor(5, 4);
$colors = [];
for ($y = 0; $y < 4; $y++) {
    for ($x = 0; $x < 5; $x++) {
        $r = ($x * 50) & 0xFF;
        $g = ($y * 60) & 0xFF;
        $b = (($x + $y) * 25) & 0xFF;
        imagesetpixel($src, $x, $y, ($r << 16) | ($g << 8) | $b);
        $colors[] = [$r, $g, $b];
    }
}

/* GPU CHW [3, H, W], transposed to HWC [H, W, 3] on the device. */
$chw = NumPower::fromImage($src, false, 'float32', NUMPOWER_CUDA);
$hwc = NumPower::transpose($chw, [1, 2, 0]);
echo 'chw_isgpu: ', ($chw->isGPU() ? 'OK' : 'BAD'), "\n";
echo 'hwc_isgpu: ', ($hwc->isGPU() ? 'OK' : 'BAD'), "\n";
echo 'chw_shape: ', json_encode($chw->shape()), "\n";
echo 'hwc_shape: ', json_encode($hwc->shape()), "\n";

$back_chw = $chw->toImage();
$back_hwc = $hwc->toImage();
$ok_chw = true;
$ok_hwc = true;
for ($y = 0; $y < 4; $y++) {
    for ($x = 0; $x < 5; $x++) {
        [$r, $g, $b] = $colors[$y * 5 + $x];
        $exp = ($r << 16) | ($g << 8) | $b;
        $got_c = imagecolorat($back_chw, $x, $y) & 0xFFFFFF;
        $got_h = imagecolorat($back_hwc, $x, $y) & 0xFFFFFF;
        if ($got_c !== $exp) $ok_chw = false;
        if ($got_h !== $exp) $ok_hwc = false;
    }
}
echo 'chw_pixels: ', ($ok_chw ? 'OK' : 'BAD'), "\n";
echo 'hwc_transposed_pixels: ', ($ok_hwc ? 'OK' : 'BAD'), "\n";

/* GPU slice: build a tall image, slice the top 3 rows, convert back.
   HWC slice has shape [3, W, 3]. With W != 3, the layout detector picks
   HWC (shape[2]==3, shape[0]!=3 needed — but here shape[0]==3, ambig…
   so HWC wins, which is correct). */
$src2 = imagecreatetruecolor(5, 6);
for ($y = 0; $y < 6; $y++) {
    for ($x = 0; $x < 5; $x++) {
        imagesetpixel($src2, $x, $y, $y * 0x101010);
    }
}
$gpu = NumPower::fromImage($src2, true, 'uint8', NUMPOWER_CUDA);
$top = NumPower::slice($gpu, [0, 3]); // [3, 5, 3]
echo 'top_isgpu: ', ($top->isGPU() ? 'OK' : 'BAD'), "\n";
$img = $top->toImage();
echo 'top_size: ', imagesx($img), 'x', imagesy($img), "\n";
$ok = true;
for ($y = 0; $y < 3; $y++) {
    $expected = $y * 0x101010;
    for ($x = 0; $x < 5; $x++) {
        if ((imagecolorat($img, $x, $y) & 0xFFFFFF) !== $expected) $ok = false;
    }
}
echo 'top_slice_pixels: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
chw_isgpu: OK
hwc_isgpu: OK
chw_shape: [3,4,5]
hwc_shape: [4,5,3]
chw_pixels: OK
hwc_transposed_pixels: OK
top_isgpu: OK
top_size: 5x3
top_slice_pixels: OK
