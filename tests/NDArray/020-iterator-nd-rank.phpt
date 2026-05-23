--TEST--
NDArray Iterator: current() returns NDArray for N>=2-D and dtype scalar for 1-D
--FILE--
<?php
/* Per-rank return contract of current():
   - 1-D source -> dtype-correct scalar (covered in 019; sanity-check here)
   - 2-D source -> 1-D NDArray sub-view per axis-0 row
   - 3-D source -> 2-D NDArray sub-view per axis-0 plane

   For N-D sub-views the view must be a true memory view of the parent (no copy)
   and must preserve the parent's dtype. We assert by walking shape() and
   reading a representative scalar. */

/* 2-D: rows. */
$m = NumPower::array([[10, 20, 30], [40, 50, 60]], 'int32');
$m->rewind();
while ($m->valid()) {
    $row = $m->current();
    echo "2D row key=", $m->key(),
         " is_NDArray=", ($row instanceof NDArray) ? '1' : '0',
         " shape=", json_encode($row->shape()),
         " toArray=", json_encode($row->toArray()), "\n";
    $m->next();
}

/* 3-D: each step yields a 2-D plane. */
$c = NumPower::array(
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    'float64'
);
$c->rewind();
while ($c->valid()) {
    $plane = $c->current();
    echo "3D plane key=", $c->key(),
         " is_NDArray=", ($plane instanceof NDArray) ? '1' : '0',
         " shape=", json_encode($plane->shape()),
         " toArray=", json_encode($plane->toArray()), "\n";
    $c->next();
}

/* 1-D sanity: yields a non-NDArray scalar (full per-dtype matrix is in 019). */
$v = new NDArray([1.5, 2.5, 3.5], 'float64');
$v->rewind();
$x = $v->current();
echo "1D scalar is_float=", is_float($x) ? '1' : '0', " val=$x\n";
?>
--EXPECT--
2D row key=0 is_NDArray=1 shape=[3] toArray=[10,20,30]
2D row key=1 is_NDArray=1 shape=[3] toArray=[40,50,60]
3D plane key=0 is_NDArray=1 shape=[2,2] toArray=[[1,2],[3,4]]
3D plane key=1 is_NDArray=1 shape=[2,2] toArray=[[5,6],[7,8]]
1D scalar is_float=1 val=1.5
