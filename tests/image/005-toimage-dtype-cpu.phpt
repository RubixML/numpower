--TEST--
NDArray::toImage() converts CHW and HWC inputs of every dtype on CPU
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* toImage roundtrip: build an NDArray from a known image, ship it back
   through toImage(), and assert every pixel decodes exactly. Tested
   for every dtype except the narrow-range float4/8 (which lose
   precision below the 0..255 pixel grid by design). */

$dtypes = ['float16','float32','float64','float128',
           'int16','uint8','uint16','int32','uint32','int64','uint64'];

/* Image with a small but non-trivial palette so 0/64/128/192/255 are
   all exercised. Non-square (W=4, H=3) so an H/W swap would surface. */
$src = imagecreatetruecolor(4, 3);
$colors = [
    0x000000, 0x404040, 0x808080, 0xC0C0C0,
    0xFF0000, 0x00FF00, 0x0000FF, 0xFFFFFF,
    0x4080C0, 0xC08040, 0x8040C0, 0x40C080,
];
for ($y = 0; $y < 3; $y++) {
    for ($x = 0; $x < 4; $x++) {
        imagesetpixel($src, $x, $y, $colors[$y * 4 + $x]);
    }
}

foreach ($dtypes as $dt) {
    /* HWC → toImage roundtrip. */
    $hwc = NumPower::fromImage($src, true, $dt);
    $img = $hwc->toImage();
    $size_ok = imagesx($img) === 4 && imagesy($img) === 3;
    $ok = true;
    for ($y = 0; $y < 3; $y++) {
        for ($x = 0; $x < 4; $x++) {
            $expect = $colors[$y * 4 + $x] & 0xFFFFFF;
            $got    = imagecolorat($img, $x, $y) & 0xFFFFFF;
            if ($got !== $expect) $ok = false;
        }
    }
    echo "$dt HWC: size=", ($size_ok ? 'OK' : 'BAD'),
         ' pixels=', ($ok ? 'OK' : 'BAD'), "\n";

    /* CHW → toImage roundtrip. */
    $chw = NumPower::fromImage($src, false, $dt);
    $img = $chw->toImage();
    $size_ok = imagesx($img) === 4 && imagesy($img) === 3;
    $ok = true;
    for ($y = 0; $y < 3; $y++) {
        for ($x = 0; $x < 4; $x++) {
            $expect = $colors[$y * 4 + $x] & 0xFFFFFF;
            $got    = imagecolorat($img, $x, $y) & 0xFFFFFF;
            if ($got !== $expect) $ok = false;
        }
    }
    echo "$dt CHW: size=", ($size_ok ? 'OK' : 'BAD'),
         ' pixels=', ($ok ? 'OK' : 'BAD'), "\n";
}

/* Float values outside [0, 255] must clamp, not corrupt adjacent
   channels by spilling into the high bytes of the packed pixel word. */
$over = NumPower::full([1, 1, 3], 500.0, 'float32');
$img = $over->toImage();
$pix = imagecolorat($img, 0, 0) & 0xFFFFFF;
echo 'clamp_high: ', ($pix === 0xFFFFFF ? 'OK' : 'BAD'), "\n";

$under = NumPower::full([1, 1, 3], -100.0, 'float32');
$img = $under->toImage();
$pix = imagecolorat($img, 0, 0) & 0xFFFFFF;
echo 'clamp_low: ', ($pix === 0x000000 ? 'OK' : 'BAD'), "\n";

/* NaN floats must round to 0 (the "missing pixel → black" convention),
   not corrupt the pixel with an undefined-behavior cast. */
$nan = NumPower::full([1, 1, 3], NAN, 'float64');
$img = $nan->toImage();
$pix = imagecolorat($img, 0, 0) & 0xFFFFFF;
echo 'clamp_nan: ', ($pix === 0x000000 ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float16 HWC: size=OK pixels=OK
float16 CHW: size=OK pixels=OK
float32 HWC: size=OK pixels=OK
float32 CHW: size=OK pixels=OK
float64 HWC: size=OK pixels=OK
float64 CHW: size=OK pixels=OK
float128 HWC: size=OK pixels=OK
float128 CHW: size=OK pixels=OK
int16 HWC: size=OK pixels=OK
int16 CHW: size=OK pixels=OK
uint8 HWC: size=OK pixels=OK
uint8 CHW: size=OK pixels=OK
uint16 HWC: size=OK pixels=OK
uint16 CHW: size=OK pixels=OK
int32 HWC: size=OK pixels=OK
int32 CHW: size=OK pixels=OK
uint32 HWC: size=OK pixels=OK
uint32 CHW: size=OK pixels=OK
int64 HWC: size=OK pixels=OK
int64 CHW: size=OK pixels=OK
uint64 HWC: size=OK pixels=OK
uint64 CHW: size=OK pixels=OK
clamp_high: OK
clamp_low: OK
clamp_nan: OK
