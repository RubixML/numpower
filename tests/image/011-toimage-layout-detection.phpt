--TEST--
NDArray::toImage() layout detection: HWC default for 3×3×3 ambiguity, transposed views work
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* The CHW vs HWC layout is determined from the shape: if `shape[2] == 3`
   the array is HWC; if `shape[0] == 3` and `shape[2] != 3` it is CHW.
   When *both* axes are 3 (a 3×3×3 cube), HWC wins — that matches the
   default of `fromImage()` and the numpy/PIL convention. This test
   pins the disambiguation rule so a future refactor cannot silently
   flip the default. */

/* Ambiguous 3×3×3 — HWC must win. */
$a = NumPower::full([3, 3, 3], 0, 'uint8');
$a[0][0][0] = 10; $a[0][0][1] = 20; $a[0][0][2] = 30;
$a[1][2][0] = 40; $a[1][2][1] = 50; $a[1][2][2] = 60;
$img = $a->toImage();
echo 'ambiguous_3x3x3_size: ', imagesx($img), 'x', imagesy($img), "\n";
$c = imagecolorat($img, 0, 0) & 0xFFFFFF;
echo 'ambiguous_3x3x3_(0,0): ', sprintf('#%06x', $c), " (HWC expects #0a141e)\n";
$c = imagecolorat($img, 2, 1) & 0xFFFFFF;
echo 'ambiguous_3x3x3_(2,1): ', sprintf('#%06x', $c), " (HWC expects #28323c)\n";

/* Unambiguous CHW with shape[0]==3 and shape[2]!=3 — channel axis 0. */
$a = NumPower::full([3, 4, 5], 0, 'uint8');
/* Set R(=channel 0) at (y=2, x=1) to 200. */
$a[0][2][1] = 200;
$img = $a->toImage();
$c = imagecolorat($img, 1, 2) & 0xFFFFFF;
echo 'chw_R_(1,2): ', sprintf('#%06x', $c), " (expect #c80000)\n";

/* Unambiguous HWC with shape[2]==3 and shape[0]!=3 — channel axis 2. */
$a = NumPower::full([4, 5, 3], 0, 'uint8');
$a[2][1][0] = 200;
$img = $a->toImage();
$c = imagecolorat($img, 1, 2) & 0xFFFFFF;
echo 'hwc_R_(1,2): ', sprintf('#%06x', $c), " (expect #c80000)\n";

/* Transposed view: build CHW, transpose to HWC, toImage must produce
   the same pixels as the original CHW. Non-contiguous strides must be
   indexed correctly. */
$src = imagecreatetruecolor(4, 3);
imagesetpixel($src, 0, 0, 0xAA0011);
imagesetpixel($src, 3, 2, 0x224488);
imagesetpixel($src, 2, 1, 0x336699);
$chw = NumPower::fromImage($src, false); // [3, 3, 4]
$hwc = NumPower::transpose($chw, [1, 2, 0]); // [3, 4, 3]
$back_chw = $chw->toImage();
$back_hwc = $hwc->toImage();
$ok = true;
for ($y = 0; $y < 3; $y++) {
    for ($x = 0; $x < 4; $x++) {
        $a_c = imagecolorat($back_chw, $x, $y) & 0xFFFFFF;
        $a_h = imagecolorat($back_hwc, $x, $y) & 0xFFFFFF;
        $src_c = imagecolorat($src, $x, $y) & 0xFFFFFF;
        if ($a_c !== $src_c || $a_h !== $src_c) $ok = false;
    }
}
echo 'transpose_roundtrip: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Sliced view: build a tall image then slice. The slice must produce
   an image of the slice's dimensions and the correct pixels. */
$src = imagecreatetruecolor(3, 6); // W=3, H=6
for ($y = 0; $y < 6; $y++) {
    for ($x = 0; $x < 3; $x++) {
        imagesetpixel($src, $x, $y, $y * 0x101010);
    }
}
$hwc = NumPower::fromImage($src); // [6, 3, 3]
$top_half = NumPower::slice($hwc, [0, 3]); // [3, 3, 3] — rows 0..2 (ambiguous, HWC)
$img = $top_half->toImage();
echo 'slice_size: ', imagesx($img), 'x', imagesy($img), " (expect 3x3)\n";
$ok = true;
for ($y = 0; $y < 3; $y++) {
    $expected = $y * 0x101010;
    $got = imagecolorat($img, 0, $y) & 0xFFFFFF;
    if ($got !== $expected) $ok = false;
}
echo 'slice_pixels: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
ambiguous_3x3x3_size: 3x3
ambiguous_3x3x3_(0,0): #0a141e (HWC expects #0a141e)
ambiguous_3x3x3_(2,1): #28323c (HWC expects #28323c)
chw_R_(1,2): #c80000 (expect #c80000)
hwc_R_(1,2): #c80000 (expect #c80000)
transpose_roundtrip: OK
slice_size: 3x3 (expect 3x3)
slice_pixels: OK
