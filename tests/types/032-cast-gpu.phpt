--TEST--
NDArray::cast() on a GPU array converts in place and stays on the GPU
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
$g = (new NDArray([1.5, 2.5, 3.5], 'float32'))->gpu();
$ret = $g->cast('int32');
echo $ret === $g ? "returns \$this\n" : "returns other\n";
echo $g->isGPU() ? "gpu\n" : "cpu\n";
echo $g->dataType(), ' ', json_encode($g->cpu()->toArray()), "\n";
?>
--EXPECT--
returns $this
gpu
int32 [1,2,3]
