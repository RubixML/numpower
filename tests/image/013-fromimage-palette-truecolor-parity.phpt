--TEST--
NumPower::fromImage() produces the same array for palette and truecolor copies of the same image
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* For images with ≤ 256 distinct colors, palette and truecolor copies
   of the same image must produce byte-identical NDArrays via
   fromImage(). The legacy palette branch passed swapped `(y, x)` args
   to `gdImagePalettePixel` — silently reading wrong indices on
   non-square images. This test pins the parity. */

$W = 5; $H = 4;

/* Build a palette image with 7 distinct colors and a known pattern.
   Use a non-square shape so any H/W swap would show up. */
$pal = imagecreate($W, $H);
$colors = [
    imagecolorallocate($pal, 255,   0,   0),
    imagecolorallocate($pal,   0, 255,   0),
    imagecolorallocate($pal,   0,   0, 255),
    imagecolorallocate($pal, 255, 255,   0),
    imagecolorallocate($pal, 255,   0, 255),
    imagecolorallocate($pal,   0, 255, 255),
    imagecolorallocate($pal, 128, 128, 128),
];
$pattern = [];
for ($y = 0; $y < $H; $y++) {
    $row = [];
    for ($x = 0; $x < $W; $x++) {
        $idx = ($x * 2 + $y * 3) % 7;
        imagesetpixel($pal, $x, $y, $colors[$idx]);
        $row[] = $idx;
    }
    $pattern[] = $row;
}

/* Build a truecolor copy of the same image (palette-converted on
   imagecopy). imagepalettetotruecolor flips the existing image. */
$tc = imagecreatetruecolor($W, $H);
imagealphablending($tc, false);
imagecopy($tc, $pal, 0, 0, 0, 0, $W, $H);

$pal_arr = NumPower::fromImage($pal);
$tc_arr  = NumPower::fromImage($tc);

echo 'pal shape: ', json_encode($pal_arr->shape()), "\n";
echo 'tc  shape: ', json_encode($tc_arr->shape()), "\n";
echo 'shape_eq: ', ($pal_arr->shape() === $tc_arr->shape() ? 'OK' : 'BAD'), "\n";

$ok = true;
$expected_rgb = [
    [255,   0,   0], [  0, 255,   0], [  0,   0, 255], [255, 255,   0],
    [255,   0, 255], [  0, 255, 255], [128, 128, 128],
];
for ($y = 0; $y < $H; $y++) {
    for ($x = 0; $x < $W; $x++) {
        $idx = $pattern[$y][$x];
        $exp = $expected_rgb[$idx];
        for ($c = 0; $c < 3; $c++) {
            if ((int)$pal_arr[$y][$x][$c] !== $exp[$c]) {
                $ok = false;
            }
            if ((int)$tc_arr[$y][$x][$c] !== $exp[$c]) {
                $ok = false;
            }
        }
    }
}
echo 'palette_truecolor_parity: ', ($ok ? 'OK' : 'BAD'), "\n";

/* CHW palette parity. */
$pal_chw = NumPower::fromImage($pal, false);
$tc_chw  = NumPower::fromImage($tc, false);
echo 'chw shape: ', json_encode($pal_chw->shape()), "\n";
$ok = true;
for ($c = 0; $c < 3; $c++) {
    for ($y = 0; $y < $H; $y++) {
        for ($x = 0; $x < $W; $x++) {
            if ((int)$pal_chw[$c][$y][$x] !== (int)$tc_chw[$c][$y][$x]) {
                $ok = false;
            }
        }
    }
}
echo 'palette_truecolor_chw_parity: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
pal shape: [4,5,3]
tc  shape: [4,5,3]
shape_eq: OK
palette_truecolor_parity: OK
chw shape: [3,4,5]
palette_truecolor_chw_parity: OK
