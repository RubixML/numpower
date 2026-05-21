--TEST--
Heavy mixed CPU/GPU/index/toArray cycles do not leak VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress the full element-access surface. 50 iterations × 14 dtypes × ~30
   ops/iter = ~21k GPU buffer create/free roundtrips. NDARRAY_VCHECK=1 at
   shutdown prints VRAM MEMORY LEAK if any cudaMalloc is unmatched. */

$types = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
          'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

for ($iter = 0; $iter < 50; $iter++) {
    foreach ($types as $t) {
        $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
        $vals1 = $strIO ? ['1','2','3','4'] : [1,2,3,4];
        $vals2 = $strIO ? [['1','2'],['3','4']] : [[1,2],[3,4]];

        $a  = new NDArray($vals1, $t);
        for ($i = 0; $i < 4; $i++) { $v = $a[$i]; }
        $cpuArr = $a->toArray();
        $g = $a->gpu();
        for ($i = 0; $i < 4; $i++) { $v = $g[$i]; }
        $back = $g->cpu();
        $backArr = $back->toArray();

        $a2 = new NDArray($vals2, $t);
        $g2 = $a2->gpu();
        $v = $g2[0][0];
        $v = $g2[1][1];

        unset($a, $g, $back, $a2, $g2, $cpuArr, $backArr, $v);
    }
}
echo "ok\n";
?>
--EXPECT--
ok
