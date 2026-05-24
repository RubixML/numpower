--TEST--
NumPower::fromImage() edge cases: 1×1, 1×N, N×1, palette images, layout consistency
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* Cover boundary geometry that would surface stride / shape bugs:
   - 1x1: smallest non-empty image; trivially asserts (H, W) ordering.
   - 1xN: H=1, W=N (one row, many columns) — exposes any H/W swap.
   - Nx1: H=N, W=1 (one column, many rows) — same, opposite axis.
   - palette mode (non-truecolor) — uses gdImagePalettePixel(x, y) which
     the legacy code called with swapped (y, x) args. */

/* 1x1 image. */
$img = imagecreatetruecolor(1, 1);
imagesetpixel($img, 0, 0, 0xAABBCC);
$a = NumPower::fromImage($img);
echo '1x1 HWC shape: ', json_encode($a->shape()), "\n";
echo '1x1 (0,0): R=', (int)$a[0][0][0], ' G=', (int)$a[0][0][1], ' B=', (int)$a[0][0][2], "\n";
$b = NumPower::fromImage($img, false);
echo '1x1 CHW shape: ', json_encode($b->shape()), "\n";

/* 1xN (1 row, 5 columns): HWC must be [1, 5, 3]. */
$img = imagecreatetruecolor(5, 1);
for ($x = 0; $x < 5; $x++) imagesetpixel($img, $x, 0, ($x * 50) << 16);
$a = NumPower::fromImage($img);
echo '1x5 HWC shape: ', json_encode($a->shape()), "\n";
$ok = true;
for ($x = 0; $x < 5; $x++) {
    if ((int)$a[0][$x][0] !== ($x * 50)) $ok = false;
}
echo '1x5 row reads: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Nx1 (5 rows, 1 column): HWC must be [5, 1, 3]. */
$img = imagecreatetruecolor(1, 5);
for ($y = 0; $y < 5; $y++) imagesetpixel($img, 0, $y, ($y * 40) << 8);
$a = NumPower::fromImage($img);
echo '5x1 HWC shape: ', json_encode($a->shape()), "\n";
$ok = true;
for ($y = 0; $y < 5; $y++) {
    if ((int)$a[$y][0][1] !== ($y * 40)) $ok = false;
}
echo '5x1 col reads: ', ($ok ? 'OK' : 'BAD'), "\n";

/* HWC/CHW value equivalence: both layouts must encode the same per-pixel
   colors, just transposed in axis order. */
$img = imagecreatetruecolor(3, 2);
imagesetpixel($img, 0, 0, 0x100000);
imagesetpixel($img, 1, 0, 0x200000);
imagesetpixel($img, 2, 0, 0x300000);
imagesetpixel($img, 0, 1, 0x001000);
imagesetpixel($img, 1, 1, 0x002000);
imagesetpixel($img, 2, 1, 0x003000);
$hwc = NumPower::fromImage($img, true);
$chw = NumPower::fromImage($img, false);
$ok = true;
for ($y = 0; $y < 2; $y++) {
    for ($x = 0; $x < 3; $x++) {
        for ($c = 0; $c < 3; $c++) {
            if ((int)$hwc[$y][$x][$c] !== (int)$chw[$c][$y][$x]) $ok = false;
        }
    }
}
echo 'HWC-CHW value parity: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Palette mode (non-truecolor): legacy code called
   gdImagePalettePixel(im, i, j) with i=y, j=x — swapped args. Use a
   non-square image so any swap shows up in the readback shape. */
$pal = imagecreate(4, 2); // palette image, W=4, H=2
$red    = imagecolorallocate($pal, 255, 0, 0);
$green  = imagecolorallocate($pal, 0, 255, 0);
$blue   = imagecolorallocate($pal, 0, 0, 255);
$white  = imagecolorallocate($pal, 255, 255, 255);
imagesetpixel($pal, 0, 0, $red);
imagesetpixel($pal, 1, 0, $green);
imagesetpixel($pal, 2, 0, $blue);
imagesetpixel($pal, 3, 0, $white);
imagesetpixel($pal, 0, 1, $white);
imagesetpixel($pal, 1, 1, $blue);
imagesetpixel($pal, 2, 1, $green);
imagesetpixel($pal, 3, 1, $red);
$a = NumPower::fromImage($pal);
echo 'palette HWC shape: ', json_encode($a->shape()), "\n";
$expected = [
    [[255,0,0], [0,255,0], [0,0,255], [255,255,255]],
    [[255,255,255], [0,0,255], [0,255,0], [255,0,0]],
];
$ok = true;
for ($y = 0; $y < 2; $y++) {
    for ($x = 0; $x < 4; $x++) {
        for ($c = 0; $c < 3; $c++) {
            if ((int)$a[$y][$x][$c] !== $expected[$y][$x][$c]) $ok = false;
        }
    }
}
echo 'palette pixel grid: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
1x1 HWC shape: [1,1,3]
1x1 (0,0): R=170 G=187 B=204
1x1 CHW shape: [3,1,1]
1x5 HWC shape: [1,5,3]
1x5 row reads: OK
5x1 HWC shape: [5,1,3]
5x1 col reads: OK
HWC-CHW value parity: OK
palette HWC shape: [2,4,3]
palette pixel grid: OK
