--TEST--
NDArray::toImage() converts GPU-resident NDArrays directly without an explicit ->cpu() call
--SKIPIF--
<?php
if (!extension_loaded('gd')) die('skip GD extension not loaded');
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The legacy toImage() rejected GPU NDArrays with "must be on CPU RAM"
   — forcing every GPU pipeline to do an explicit `->cpu()`. This test
   exercises the auto-staging via NDArray_TypedD2H for every dtype and
   both layouts, and confirms the pixel grid round-trips byte-for-byte. */

$dtypes = ['float16','float32','float64','float128',
           'int16','uint8','uint16','int32','uint32','int64','uint64'];

/* W=5, H=4 (non-square, no axis equal to 3 — keeps CHW vs HWC
   detection unambiguous: shape[0]==3 only matches CHW, shape[2]==3
   only matches HWC). */
$src = imagecreatetruecolor(5, 4);
$colors = [
    0x100000, 0x001000, 0x000010, 0xC0C0C0, 0xFFFFFF,
    0x202020, 0x808080, 0x000000, 0xAABBCC, 0x123456,
    0xFFAA00, 0x00AAFF, 0xAA00FF, 0x004080, 0x804000,
    0x408000, 0xFF00AA, 0x00FFAA, 0xAAFF00, 0x123456,
];
for ($y = 0; $y < 4; $y++) {
    for ($x = 0; $x < 5; $x++) {
        imagesetpixel($src, $x, $y, $colors[$y * 5 + $x]);
    }
}

foreach ($dtypes as $dt) {
    foreach ([true, false] as $cl) {
        $g = NumPower::fromImage($src, $cl, $dt, NUMPOWER_CUDA);
        $back = $g->toImage();
        $size_ok = imagesx($back) === 5 && imagesy($back) === 4;
        $ok = true;
        for ($y = 0; $y < 4; $y++) {
            for ($x = 0; $x < 5; $x++) {
                $expect = $colors[$y * 5 + $x] & 0xFFFFFF;
                $got    = imagecolorat($back, $x, $y) & 0xFFFFFF;
                if ($got !== $expect) $ok = false;
            }
        }
        $label = $cl ? 'HWC' : 'CHW';
        echo "$dt $label: size=", ($size_ok ? 'OK' : 'BAD'),
             ' pixels=', ($ok ? 'OK' : 'BAD'), "\n";
    }
}

/* Mixed-device alpha: image on GPU, alpha on CPU (and vice versa) must
   both work, because each operand is staged independently. */
$gpu_img = NumPower::fromImage($src, true, 'uint8', NUMPOWER_CUDA);
$cpu_alpha = NumPower::full([4, 5], 200, 'uint8');
$with = $gpu_img->toImage($cpu_alpha);
$pix = imagecolorat($with, 0, 0);
echo 'mixed_gpu_image_cpu_alpha_alpha_byte: ',
     ((($pix >> 24) & 0xFF) === 200 ? 'OK' : 'BAD'), "\n";

$cpu_img = NumPower::fromImage($src);
$gpu_alpha = NumPower::full([4, 5], 100, 'float32', NUMPOWER_CUDA);
$with2 = $cpu_img->toImage($gpu_alpha);
$pix2 = imagecolorat($with2, 0, 0);
echo 'mixed_cpu_image_gpu_alpha_alpha_byte: ',
     ((($pix2 >> 24) & 0xFF) === 100 ? 'OK' : 'BAD'), "\n";
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
mixed_gpu_image_cpu_alpha_alpha_byte: OK
mixed_cpu_image_gpu_alpha_alpha_byte: OK
