--TEST--
Repeated NDArray::fill() on GPU NDArrays does not leak VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* fill() on a GPU array allocates a host-side typed buffer, writes the
   scalar into every cell, hands it to TypedH2D, and frees the host buffer.
   Hammer the path on every dtype + every supported PHP scalar type to catch
   any unmatched cudaMalloc / cudaFree pair or efree-skipping error branch. */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

for ($iter = 0; $iter < 50; $iter++) {
    foreach ($types as $t) {
        $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $vals = $strIO ? ['1','2','3','4'] : [1,2,3,4];
        $a = (new NDArray($vals, $t))->gpu();

        $a->fill('3');
        $a->fill(1);
        $a->fill(2.0);
        $a->fill(true);
        $a->fill(false);

        unset($a);
    }
}
echo "ok\n";
?>
--EXPECT--
ok
