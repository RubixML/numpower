--TEST--
NDArray::toImage() error paths: bad shapes, alpha mismatches, and alpha round-trip
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* Validation gates must throw a catchable \Error and never segfault.
   Covers: non-3-D NDArrays, 3-D NDArrays where no axis is 3, alpha
   shape mismatches. Also verifies that an alpha round-trip preserves
   the alpha byte exactly. */

/* 2-D NDArray → reject (image must be 3-D). */
$two_d = NumPower::full([4, 4], 100, 'uint8');
try { $two_d->toImage(); echo "2d: BAD (no throw)\n"; }
catch (\Error $e) { echo "2d: OK\n"; }

/* 4-D → reject. */
$four_d = NumPower::full([3, 4, 5, 3], 100, 'uint8');
try { $four_d->toImage(); echo "4d: BAD (no throw)\n"; }
catch (\Error $e) { echo "4d: OK\n"; }

/* 3-D but no axis of size 3 → reject. */
$wrong = NumPower::full([4, 5, 6], 100, 'uint8');
try { $wrong->toImage(); echo "no_channel_axis: BAD (no throw)\n"; }
catch (\Error $e) { echo "no_channel_axis: OK\n"; }

/* Alpha shape mismatch. */
$img = NumPower::full([3, 4, 5], 128, 'uint8'); // CHW: 3 channels, H=4, W=5
$bad_alpha = NumPower::full([3, 3], 200, 'uint8');
try { $img->toImage($bad_alpha); echo "alpha_shape: BAD (no throw)\n"; }
catch (\Error $e) { echo "alpha_shape: OK\n"; }

/* Alpha not 2-D — reject. */
$alpha_1d = NumPower::full([4], 200, 'uint8');
try { $img->toImage($alpha_1d); echo "alpha_ndim: BAD (no throw)\n"; }
catch (\Error $e) { echo "alpha_ndim: OK\n"; }

/* Alpha round-trip: every pixel's high byte must equal the alpha
   array element after toImage. The legacy alpha indexing used the 3-D
   image strides for the 2-D alpha array — looking far out of bounds
   after the first row. This test would have caught that. */
$src = imagecreatetruecolor(5, 4);
for ($y = 0; $y < 4; $y++)
    for ($x = 0; $x < 5; $x++)
        imagesetpixel($src, $x, $y, ($y*60 << 16) | ($x*50 << 8));
$hwc = NumPower::fromImage($src);
/* Per-pixel alpha gradient that varies in BOTH dimensions so a wrong
   stride would mismatch any row beyond the first. */
$alpha_php = [];
for ($y = 0; $y < 4; $y++) {
    $row = [];
    for ($x = 0; $x < 5; $x++) {
        $row[] = $x * 50 + $y * 10;
    }
    $alpha_php[] = $row;
}
$alpha = NumPower::array($alpha_php, 'uint8');
$rgba = $hwc->toImage($alpha);
$ok = true;
for ($y = 0; $y < 4; $y++) {
    for ($x = 0; $x < 5; $x++) {
        $expected_alpha = $x * 50 + $y * 10;
        $color = imagecolorat($rgba, $x, $y);
        $actual_alpha = ($color >> 24) & 0xFF;
        if ($actual_alpha !== $expected_alpha) {
            $ok = false;
        }
    }
}
echo 'alpha_gradient_roundtrip: ', ($ok ? 'OK' : 'BAD'), "\n";

/* Alpha clamps: values > 255 saturate to 255, < 0 to 0. */
$over = NumPower::full([4, 5], 1000.0, 'float32');
$rgba = $hwc->toImage($over);
$pix = imagecolorat($rgba, 2, 2);
echo 'alpha_clamp_high: ', ((($pix >> 24) & 0xFF) === 255 ? 'OK' : 'BAD'), "\n";

$under = NumPower::full([4, 5], -50.0, 'float32');
$rgba = $hwc->toImage($under);
$pix = imagecolorat($rgba, 2, 2);
echo 'alpha_clamp_low: ', ((($pix >> 24) & 0xFF) === 0 ? 'OK' : 'BAD'), "\n";

/* Null alpha (explicit) — RGB image with alpha byte == 0. */
$rgb = $hwc->toImage(null);
$pix = imagecolorat($rgb, 0, 0);
echo 'null_alpha: ', ((($pix >> 24) & 0xFF) === 0 ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
2d: OK
4d: OK
no_channel_axis: OK
alpha_shape: OK
alpha_ndim: OK
alpha_gradient_roundtrip: OK
alpha_clamp_high: OK
alpha_clamp_low: OK
null_alpha: OK
