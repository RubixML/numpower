--TEST--
NumPower::fromImage() handles larger geometries and boundary pixel values
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* Larger geometries (64×48, 100×80) and boundary pixel values (0, 1,
   127, 128, 254, 255). These pin the per-row stride math against
   off-by-one bugs that wouldn't surface on tiny test images. */

$boundary_values = [0, 1, 127, 128, 254, 255];

/* 64×48 image whose pixels cycle through the boundary values for R,
   G, B independently. */
$W = 64; $H = 48;
$img = imagecreatetruecolor($W, $H);
for ($y = 0; $y < $H; $y++) {
    for ($x = 0; $x < $W; $x++) {
        $r = $boundary_values[($x + $y) % 6];
        $g = $boundary_values[($x * 2 + $y) % 6];
        $b = $boundary_values[($y * 3 + $x) % 6];
        imagesetpixel($img, $x, $y, ($r << 16) | ($g << 8) | $b);
    }
}

/* uint8 round-trip: every pixel byte should be preserved exactly. */
$a = NumPower::fromImage($img);
echo 'shape: ', json_encode($a->shape()), "\n";
$ok = true;
for ($y = 0; $y < $H; $y++) {
    for ($x = 0; $x < $W; $x++) {
        $r = $boundary_values[($x + $y) % 6];
        $g = $boundary_values[($x * 2 + $y) % 6];
        $b = $boundary_values[($y * 3 + $x) % 6];
        if ((int)$a[$y][$x][0] !== $r ||
            (int)$a[$y][$x][1] !== $g ||
            (int)$a[$y][$x][2] !== $b) {
            $ok = false;
        }
    }
}
echo '64x48_uint8_grid: ', ($ok ? 'OK' : 'BAD'), "\n";

/* CHW variant of the same image. */
$c = NumPower::fromImage($img, false);
echo 'CHW shape: ', json_encode($c->shape()), "\n";
$ok = true;
for ($y = 0; $y < $H; $y++) {
    for ($x = 0; $x < $W; $x++) {
        $r = $boundary_values[($x + $y) % 6];
        $g = $boundary_values[($x * 2 + $y) % 6];
        $b = $boundary_values[($y * 3 + $x) % 6];
        if ((int)$c[0][$y][$x] !== $r ||
            (int)$c[1][$y][$x] !== $g ||
            (int)$c[2][$y][$x] !== $b) {
            $ok = false;
        }
    }
}
echo '64x48_CHW_grid: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Roundtrip through toImage on the larger image — every byte preserved. */
$back = $a->toImage();
$ok = true;
for ($y = 0; $y < $H; $y++) {
    for ($x = 0; $x < $W; $x++) {
        $expect = (imagecolorat($img, $x, $y) & 0xFFFFFF);
        $got    = (imagecolorat($back, $x, $y) & 0xFFFFFF);
        if ($got !== $expect) $ok = false;
    }
}
echo '64x48_roundtrip: ', ($ok ? 'OK' : 'BAD'), "\n";

/* 100×80 with the largest pixel-row strides we test; assert the last
   column / last row aren't corrupted (off-by-one in stride math). */
$W = 100; $H = 80;
$img = imagecreatetruecolor($W, $H);
imagesetpixel($img, 0, 0, 0x111111);
imagesetpixel($img, $W - 1, 0, 0x222222);
imagesetpixel($img, 0, $H - 1, 0x333333);
imagesetpixel($img, $W - 1, $H - 1, 0x444444);
$a = NumPower::fromImage($img);
echo 'corner_tl: ', ((int)$a[0][0][0] === 0x11 ? 'OK' : 'BAD'), "\n";
echo 'corner_tr: ', ((int)$a[0][$W - 1][0] === 0x22 ? 'OK' : 'BAD'), "\n";
echo 'corner_bl: ', ((int)$a[$H - 1][0][0] === 0x33 ? 'OK' : 'BAD'), "\n";
echo 'corner_br: ', ((int)$a[$H - 1][$W - 1][0] === 0x44 ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
shape: [48,64,3]
64x48_uint8_grid: OK
CHW shape: [3,48,64]
64x48_CHW_grid: OK
64x48_roundtrip: OK
corner_tl: OK
corner_tr: OK
corner_bl: OK
corner_br: OK
