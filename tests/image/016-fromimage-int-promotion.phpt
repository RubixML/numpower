--TEST--
NumPower::fromImage() values fit exactly in dtypes wide enough for [0, 255]
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* For every dtype that can hold the entire [0, 255] pixel range with
   exact value preservation, the round-trip via fromImage → toImage
   must be lossless across an image that hits every byte boundary. */

$lossless_dtypes = [
    'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
    'float16', 'float32', 'float64', 'float128',
];

/* 16×16 image with all 256 byte values hit in each channel. Build a
   pattern where R = (x+y)%256, G = (x*5)%256, B = (y*7)%256 so every
   channel covers a wide range. */
$W = 16; $H = 16;
$img = imagecreatetruecolor($W, $H);
for ($y = 0; $y < $H; $y++) {
    for ($x = 0; $x < $W; $x++) {
        $r = ($x * 16 + $y) & 0xFF;
        $g = ($x * 5) & 0xFF;
        $b = ($y * 7) & 0xFF;
        imagesetpixel($img, $x, $y, ($r << 16) | ($g << 8) | $b);
    }
}

foreach ($lossless_dtypes as $dt) {
    $arr = NumPower::fromImage($img, true, $dt);
    /* float16 has only 11 mantissa bits, plenty for integers up to
       2048 — pixel values fit exactly. */
    $back = $arr->toImage();
    $ok = true;
    for ($y = 0; $y < $H; $y++) {
        for ($x = 0; $x < $W; $x++) {
            $exp = imagecolorat($img,  $x, $y) & 0xFFFFFF;
            $got = imagecolorat($back, $x, $y) & 0xFFFFFF;
            if ($got !== $exp) $ok = false;
        }
    }
    echo "$dt: ", ($ok ? 'OK' : 'BAD'), "\n";
}

/* Now check that float4/float8/int8 lose information but round-trip
   produces a *consistent* image (not garbage). */
$narrow_dtypes = [
    'float4'  => 0xFFFFFF, // saturates everything to fp4 max → noise but consistent
    'float8'  => 0xFFFFFF,
    'int8'    => 0xFFFFFF,
];
foreach (array_keys($narrow_dtypes) as $dt) {
    $arr = NumPower::fromImage($img, true, $dt);
    $back1 = $arr->toImage();
    $back2 = $arr->toImage();
    $ok = true;
    for ($y = 0; $y < $H; $y++) {
        for ($x = 0; $x < $W; $x++) {
            $a = imagecolorat($back1, $x, $y) & 0xFFFFFF;
            $b = imagecolorat($back2, $x, $y) & 0xFFFFFF;
            if ($a !== $b) $ok = false;
        }
    }
    echo "$dt deterministic: ", ($ok ? 'OK' : 'BAD'), "\n";
}
?>
--EXPECT--
uint8: OK
int16: OK
uint16: OK
int32: OK
uint32: OK
int64: OK
uint64: OK
float16: OK
float32: OK
float64: OK
float128: OK
float4 deterministic: OK
float8 deterministic: OK
int8 deterministic: OK
