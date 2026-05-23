--TEST--
NDArray Iterator: GPU 2-D and 3-D sources yield GPU sub-views (no host copy)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Iterating an N-D GPU NDArray must hand back GPU sub-views, not silently
   demote to CPU. Each row/plane keeps isGPU()==true and its data lives in
   VRAM until the user explicitly cpu()-s it. */

$m = (new NDArray([[10, 20, 30], [40, 50, 60]], 'int32'))->gpu();
$m->rewind();
while ($m->valid()) {
    $row = $m->current();
    echo "2D row key=", $m->key(),
         " is_NDArray=", ($row instanceof NDArray) ? '1' : '0',
         " isGPU=", $row->isGPU() ? '1' : '0',
         " shape=", json_encode($row->shape()),
         " toArray=", json_encode($row->cpu()->toArray()), "\n";
    $m->next();
}

$c = (new NDArray(
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    'float64'))->gpu();
$c->rewind();
while ($c->valid()) {
    $plane = $c->current();
    echo "3D plane key=", $c->key(),
         " is_NDArray=", ($plane instanceof NDArray) ? '1' : '0',
         " isGPU=", $plane->isGPU() ? '1' : '0',
         " shape=", json_encode($plane->shape()),
         " toArray=", json_encode($plane->cpu()->toArray()), "\n";
    $c->next();
}
?>
--EXPECT--
2D row key=0 is_NDArray=1 isGPU=1 shape=[3] toArray=[10,20,30]
2D row key=1 is_NDArray=1 isGPU=1 shape=[3] toArray=[40,50,60]
3D plane key=0 is_NDArray=1 isGPU=1 shape=[2,2] toArray=[[1,2],[3,4]]
3D plane key=1 is_NDArray=1 isGPU=1 shape=[2,2] toArray=[[5,6],[7,8]]
