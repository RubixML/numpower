--TEST--
Repeated offsetSet on GPU NDArrays does not leak VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* offsetSet on a GPU array allocates a host-side typed buffer, fills it,
   cudaMemcpys to GPU, then frees the host buffer. Stress this path to
   catch any unmatched cudaMalloc / cudaFree pair. */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

for ($iter = 0; $iter < 50; $iter++) {
    foreach ($types as $t) {
        $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $vals = $strIO ? ['1','2','3','4'] : [1,2,3,4];
        $a = (new NDArray($vals, $t))->gpu();
        for ($i = 0; $i < 4; $i++) {
            if ($t === 'float128' || $t === 'uint64') {
                $a[$i] = '7';
            } else {
                $a[$i] = 7;
            }
        }
        unset($a);
    }
}
echo "ok\n";
?>
--EXPECT--
ok
