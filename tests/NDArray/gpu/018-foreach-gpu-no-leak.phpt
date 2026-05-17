--TEST--
foreach over GPU NDArrays does not leak VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* PHP_METHOD(NDArray, current) creates a per-iteration slice. For 1-D source
   that slice is 0-dim and gets freed by NDArray_ScalarToZval's caller. This
   test repeats foreach 100× over every dtype to confirm the slice cleanup
   path doesn't leak GPU buffers. */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

for ($iter = 0; $iter < 100; $iter++) {
    foreach ($types as $t) {
        $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $vals = $strIO ? ['1','2','3','4'] : [1,2,3,4];
        $a = new NDArray($vals, $t);
        $g = $a->gpu();
        foreach ($g as $k => $v) {
            /* discard */
        }
        unset($a, $g);
    }
}
echo "ok\n";
?>
--EXPECT--
ok
