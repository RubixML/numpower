--TEST--
fromImage()-built GPU arrays integrate with arithmetic without staging through CPU
--SKIPIF--
<?php
if (!extension_loaded('gd')) die('skip GD extension not loaded');
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The whole point of building directly into VRAM is so the result can
   feed into GPU arithmetic without a round-trip through the host. This
   test confirms:
   - normalization (x / 255) on a float32 GPU image stays on GPU and
     produces values in [0, 1];
   - on-device add stays on GPU and produces the expected sum;
   - toImage stages the result back, with values clamped properly. */

$img = imagecreatetruecolor(8, 6);
for ($y = 0; $y < 6; $y++) {
    for ($x = 0; $x < 8; $x++) {
        imagesetpixel($img, $x, $y, ($x * 32) << 16);
    }
}

/* Normalize the GPU float32 image to [0, 1] without going back to CPU. */
$f = NumPower::fromImage($img, true, 'float32', NUMPOWER_CUDA);
$norm = NumPower::divide($f, 255.0);
echo 'norm_isgpu: ', ($norm->isGPU() ? 'OK' : 'BAD'), "\n";
$max = NumPower::max($norm);
echo 'norm_max_le_1: ', ((float)(string)$max <= 1.0 ? 'OK' : 'BAD'), "\n";
$min = NumPower::min($norm);
echo 'norm_min_ge_0: ', ((float)(string)$min >= 0.0 ? 'OK' : 'BAD'), "\n";

/* On-device add: keep on GPU, sum stays in VRAM, decoded values double. */
$sum = NumPower::add($f, $f);
echo 'sum_isgpu: ', ($sum->isGPU() ? 'OK' : 'BAD'), "\n";
/* `f` at (y=0, x=7) is R = 7*32 = 224, sum_R = 448. */
$cpu = $sum->cpu();
echo 'sum_at_(0,7,0)=448: ', ((int)$cpu[0][7][0] === 448 ? 'OK' : 'BAD'), "\n";

/* Cast the float32 sum back into a uint8 image via toImage; values
   above 255 must clamp, not wrap. */
$back = $sum->toImage();
$pix = imagecolorat($back, 7, 0) & 0xFFFFFF;
echo 'clamped_pix_r: ', ((($pix >> 16) & 0xFF) === 255 ? 'OK' : 'BAD'), "\n";

/* Pure GPU roundtrip: build → toImage → back to uint8 NDArray on GPU. */
$cpu_img = NumPower::fromImage($img, true, 'uint8', NUMPOWER_CUDA);
$back = $cpu_img->toImage();
$rebuilt = NumPower::fromImage($back, true, 'uint8', NUMPOWER_CUDA);
$diff = NumPower::subtract($cpu_img, $rebuilt);
$abs_max = (int)(string)NumPower::max(NumPower::abs($diff));
echo 'gpu_roundtrip_lossless: ', ($abs_max === 0 ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
norm_isgpu: OK
norm_max_le_1: OK
norm_min_ge_0: OK
sum_isgpu: OK
sum_at_(0,7,0)=448: OK
clamped_pix_r: OK
gpu_roundtrip_lossless: OK
