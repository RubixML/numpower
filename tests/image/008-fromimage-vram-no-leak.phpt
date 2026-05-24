--TEST--
NumPower::fromImage() / NDArray::toImage() do not leak VRAM across many allocations
--SKIPIF--
<?php
if (!extension_loaded('gd')) die('skip GD extension not loaded');
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress test: many fromImage+toImage cycles across every dtype + a mix
   of geometries (1×1, 16×9, 32×24). Each iteration's NDArray is
   overwritten — the previous VRAM slot must be released. Any imbalance
   surfaces as `VRAM MEMORY LEAK: leaked N array(s)` at RSHUTDOWN. */

$dtypes = ['float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$geoms = [[1, 1], [16, 9], [32, 24]];

foreach ($geoms as [$W, $H]) {
    $img = imagecreatetruecolor($W, $H);
    for ($y = 0; $y < $H; $y++) {
        for ($x = 0; $x < $W; $x++) {
            imagesetpixel($img, $x, $y, (($x * 7 + $y * 11) * 0x010101) & 0xFFFFFF);
        }
    }
    foreach ($dtypes as $dt) {
        foreach ([true, false] as $cl) {
            for ($i = 0; $i < 3; $i++) {
                $a = NumPower::fromImage($img, $cl, $dt, NUMPOWER_CUDA);
                /* Force a D2H readback to keep the staging buffer path
                   exercised on every iteration. */
                if ($W > 0 && $H > 0) {
                    $cpu = $a->cpu();
                    $cpu = null;
                }
                /* Also exercise toImage GPU path: the D2H staging here
                   must release. */
                $img2 = $a->toImage();
                $img2 = null;
                $a = null;
            }
        }
    }
}

echo "done\n";
?>
--EXPECT--
done
