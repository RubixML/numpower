--TEST--
NumPower::fromImage($img, $cl, $dtype, NUMPOWER_CUDA) builds directly in VRAM for every dtype
--SKIPIF--
<?php
if (!extension_loaded('gd')) die('skip GD extension not loaded');
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU build: pixel data is staged through `NDArray_TypedH2D`, which
   converts fp128 to the on-device DD layout. The result must:
   - report isGPU == true;
   - decode to the same CPU values as the CPU build (CPU/GPU parity).
   This pins the contract that the GPU storage is a faithful copy and
   no per-pixel host bytes leak through after the upload. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$img = imagecreatetruecolor(5, 4); // W=5, H=4
$colors = [];
for ($y = 0; $y < 4; $y++) {
    for ($x = 0; $x < 5; $x++) {
        $r = ($x * 50)         % 256;
        $g = ($x * 30 + $y*10) % 256;
        $b = ($y * 60)         % 256;
        $colors[] = [$r, $g, $b];
        imagesetpixel($img, $x, $y, ($r << 16) | ($g << 8) | $b);
    }
}

foreach ($dtypes as $dt) {
    foreach ([true, false] as $channel_last) {
        $gpu = NumPower::fromImage($img, $channel_last, $dt, NUMPOWER_CUDA);
        $cpu = NumPower::fromImage($img, $channel_last, $dt);
        $back = $gpu->cpu();
        $gpu_ok    = $gpu->isGPU();
        $shape_ok  = $gpu->shape() === $cpu->shape();
        $values_ok = (string)$back === (string)$cpu;
        $label = $channel_last ? 'HWC' : 'CHW';
        echo "$dt $label: gpu=", ($gpu_ok ? 1 : 0),
             ' shape=', ($shape_ok ? 'OK' : 'BAD'),
             ' values=', ($values_ok ? 'OK' : 'BAD'),
             "\n";
    }
}

/* GPU + on-device arithmetic stays on GPU: add a uint8 GPU image to
   itself; result stays on GPU and decodes to 2× per channel. The
   uint8 wrap (255 + 255 → 254 = 510 mod 256) confirms the integer
   wrap path is exercised, not float fallback. */
$g = NumPower::fromImage($img, true, 'uint8', NUMPOWER_CUDA);
$sum = NumPower::add($g, $g);
echo 'gpu_arith_isgpu=', ($sum->isGPU() ? 'OK' : 'BAD'), "\n";
$sum_cpu = $sum->cpu();
$ok = true;
for ($y = 0; $y < 4; $y++) {
    for ($x = 0; $x < 5; $x++) {
        [$r, $g0, $b] = $colors[$y * 5 + $x];
        $expect = [($r*2) & 0xFF, ($g0*2) & 0xFF, ($b*2) & 0xFF];
        $got = [(int)$sum_cpu[$y][$x][0], (int)$sum_cpu[$y][$x][1], (int)$sum_cpu[$y][$x][2]];
        if ($got !== $expect) $ok = false;
    }
}
echo 'gpu_arith_values=', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float4 HWC: gpu=1 shape=OK values=OK
float4 CHW: gpu=1 shape=OK values=OK
float8 HWC: gpu=1 shape=OK values=OK
float8 CHW: gpu=1 shape=OK values=OK
float16 HWC: gpu=1 shape=OK values=OK
float16 CHW: gpu=1 shape=OK values=OK
float32 HWC: gpu=1 shape=OK values=OK
float32 CHW: gpu=1 shape=OK values=OK
float64 HWC: gpu=1 shape=OK values=OK
float64 CHW: gpu=1 shape=OK values=OK
float128 HWC: gpu=1 shape=OK values=OK
float128 CHW: gpu=1 shape=OK values=OK
int8 HWC: gpu=1 shape=OK values=OK
int8 CHW: gpu=1 shape=OK values=OK
uint8 HWC: gpu=1 shape=OK values=OK
uint8 CHW: gpu=1 shape=OK values=OK
int16 HWC: gpu=1 shape=OK values=OK
int16 CHW: gpu=1 shape=OK values=OK
uint16 HWC: gpu=1 shape=OK values=OK
uint16 CHW: gpu=1 shape=OK values=OK
int32 HWC: gpu=1 shape=OK values=OK
int32 CHW: gpu=1 shape=OK values=OK
uint32 HWC: gpu=1 shape=OK values=OK
uint32 CHW: gpu=1 shape=OK values=OK
int64 HWC: gpu=1 shape=OK values=OK
int64 CHW: gpu=1 shape=OK values=OK
uint64 HWC: gpu=1 shape=OK values=OK
uint64 CHW: gpu=1 shape=OK values=OK
gpu_arith_isgpu=OK
gpu_arith_values=OK
