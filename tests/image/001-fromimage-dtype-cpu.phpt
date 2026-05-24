--TEST--
NumPower::fromImage() covers every supported dtype on CPU and emits the [H, W, 3] HWC shape
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* fromImage($img) must produce a 3-D RGB NDArray for every dtype, with:
   - shape [H, W, 3] for HWC (the default — was [W, H, 3] before the
     fix), shape [3, H, W] for CHW;
   - one channel value per pixel encoded in the dtype's representation
     (uint8 by default — pixel values are 0..255);
   - the same numeric values reachable through indexing as PyTorch /
     PIL would expose them. Pixel order: row-major as drawn. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$expected_type = [
    'float4'   => 'double',  'float8'   => 'double',  'float16'  => 'double',
    'float32'  => 'double',  'float64'  => 'double',  'float128' => 'string',
    'int8'     => 'integer', 'uint8'    => 'integer',
    'int16'    => 'integer', 'uint16'   => 'integer',
    'int32'    => 'integer', 'uint32'   => 'integer',
    'int64'    => 'integer', 'uint64'   => 'string',
];

/* Build a 4x3 image (W=4, H=3) with a known per-pixel RGB pattern. The
   non-square shape exposes any H/W swap; the legacy code emitted
   [W, H, 3] which would fail this. */
$img = imagecreatetruecolor(4, 3);
$colors = [
    [0xFF, 0x00, 0x00], [0x00, 0xFF, 0x00], [0x00, 0x00, 0xFF], [0x80, 0x80, 0x80],
    [0xC0, 0x10, 0x20], [0x10, 0xC0, 0x20], [0x20, 0x10, 0xC0], [0xFE, 0xFD, 0xFC],
    [0x00, 0x00, 0x00], [0xFF, 0xFF, 0xFF], [0x40, 0x80, 0xC0], [0x01, 0x02, 0x03],
];
for ($y = 0; $y < 3; $y++) {
    for ($x = 0; $x < 4; $x++) {
        [$r, $g, $b] = $colors[$y * 4 + $x];
        imagesetpixel($img, $x, $y, ($r << 16) | ($g << 8) | $b);
    }
}

/* HWC: shape [H, W, 3] for every dtype. */
foreach ($dtypes as $dt) {
    $a = NumPower::fromImage($img, true, $dt);
    $shape_ok = $a->shape() === [3, 4, 3];
    $r00 = $a[0][0][0];
    $g00 = $a[0][0][1];
    $b00 = $a[0][0][2];

    /* For each dtype the [0,0] pixel red value (255) cast through the
       dtype's representation:
         - int8        — 255 wraps to -1   (two's-complement narrowing).
         - float4/8    — dynamic range narrower than 255; the value
                         saturates / loses precision (255→6 for fp4 with
                         a ~6.0 max, 255→240 for fp8/E5M2).
         - everything else — preserves 255 exactly. */
    $expected_r = match ($dt) {
        'int8'   => -1,
        'float4' => 6,
        'float8' => 240,
        default  => 255,
    };
    $type_ok = gettype($r00) === $expected_type[$dt];
    $r_ok = ($r00 == (string)$expected_r) || ((int)$r00 === $expected_r) ||
            ((float)$r00 === (float)$expected_r);
    $g_ok = ($g00 == '0') || ((int)$g00 === 0);
    $b_ok = ($b00 == '0') || ((int)$b00 === 0);
    echo $dt, ': shape=', ($shape_ok ? 'OK' : 'BAD'),
         ' type=', ($type_ok ? 'OK' : 'BAD'),
         ' (0,0)R=', ($r_ok ? 'OK' : 'BAD'),
         ' G=', ($g_ok ? 'OK' : 'BAD'),
         ' B=', ($b_ok ? 'OK' : 'BAD'),
         ' device=', ($a->isGPU() ? 'BAD' : 'OK'),
         "\n";
}

/* CHW: shape [3, H, W]. */
foreach ($dtypes as $dt) {
    $a = NumPower::fromImage($img, false, $dt);
    $shape_ok = $a->shape() === [3, 3, 4];
    echo $dt, ' CHW: shape=', ($shape_ok ? 'OK' : 'BAD'), "\n";
}

/* Default dtype must be uint8 (the most appropriate fit for 0..255). */
$def = NumPower::fromImage($img);
echo 'default_dtype_is_uint8: ', (gettype($def[0][0][0]) === 'integer' && $def[0][0][0] === 255 ? 'OK' : 'BAD'), "\n";

/* HWC indexing convention: arr[y][x][c] reads pixel at (x, y) channel c.
   Sanity-check the entire 4x3 grid against the expected colors. */
$a = NumPower::fromImage($img);
$ok = true;
for ($y = 0; $y < 3; $y++) {
    for ($x = 0; $x < 4; $x++) {
        [$r, $g, $b] = $colors[$y * 4 + $x];
        if ((int)$a[$y][$x][0] !== $r ||
            (int)$a[$y][$x][1] !== $g ||
            (int)$a[$y][$x][2] !== $b) {
            $ok = false;
        }
    }
}
echo 'HWC_pixel_grid: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float4: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
float8: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
float16: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
float32: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
float64: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
float128: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
int8: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
uint8: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
int16: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
uint16: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
int32: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
uint32: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
int64: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
uint64: shape=OK type=OK (0,0)R=OK G=OK B=OK device=OK
float4 CHW: shape=OK
float8 CHW: shape=OK
float16 CHW: shape=OK
float32 CHW: shape=OK
float64 CHW: shape=OK
float128 CHW: shape=OK
int8 CHW: shape=OK
uint8 CHW: shape=OK
int16 CHW: shape=OK
uint16 CHW: shape=OK
int32 CHW: shape=OK
uint32 CHW: shape=OK
int64 CHW: shape=OK
uint64 CHW: shape=OK
default_dtype_is_uint8: OK
HWC_pixel_grid: OK
