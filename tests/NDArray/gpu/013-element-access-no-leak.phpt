--TEST--
GPU element access ($a[i]) does not leak device buffers
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Repeatedly access every element of every-dtype arrays on GPU. The shutdown
   hook should not print "VRAM MEMORY LEAK" — if it does, our slice-creation
   path is allocating GPU buffers it never frees. */

$types = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
          'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($types as $t) {
    $vals = (str_starts_with($t, 'float') || $t === 'uint64')
        ? ['1', '2', '3', '4']
        : [1, 2, 3, 4];
    $cpu = new NDArray($vals, $t);
    $gpu = $cpu->gpu();
    for ($i = 0; $i < 4; $i++) {
        /* discard */
        $v = $gpu[$i];
    }
    $back = $gpu->cpu();
    unset($cpu, $gpu, $back, $v);
}
echo "ok\n";
?>
--EXPECT--
ok
