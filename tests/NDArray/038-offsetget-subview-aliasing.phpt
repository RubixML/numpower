--TEST--
NDArray::offsetGet() — N-D sub-view aliases the parent buffer (writes propagate)
--FILE--
<?php
/* offsetGet on N-D source returns a sub-view whose data pointer is offset
   into the parent's buffer (rtn->base = parent, rtn->data = parent->data +
   stride*i). This is the key zero-copy property — writes to the sub-view
   must propagate to the parent, and the sub-view must remain valid for the
   parent's lifetime (the ADDREF inside iterator_view_at keeps the parent
   alive even if PHP unsets it). Verified for CPU dtypes that support
   in-place scalar broadcast through offsetSet. */

$cases = [
    ['float32', [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                7.0,    [[7.0, 7.0, 7.0], [4.0, 5.0, 6.0]]],
    ['float64', [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                7.0,    [[7.0, 7.0, 7.0], [4.0, 5.0, 6.0]]],
    ['int32',   [[1, 2, 3], [4, 5, 6]],
                7,      [[7, 7, 7], [4, 5, 6]]],
    ['int64',   [[1, 2, 3], [4, 5, 6]],
                PHP_INT_MAX, [[PHP_INT_MAX, PHP_INT_MAX, PHP_INT_MAX], [4, 5, 6]]],
];

foreach ($cases as [$dtype, $init, $fill, $expected]) {
    $a = new NDArray($init, $dtype);
    /* Sub-view aliases row 0 of $a. */
    $row = $a[0];
    $row[0] = $fill;
    $row[1] = $fill;
    $row[2] = $fill;
    $got = $a->toArray();
    echo "$dtype: ", ($got == $expected ? "OK" : "BAD got=" . json_encode($got)), "\n";
}

/* Parent survival: unset the parent PHP variable, keep the sub-view alive,
   read from it. The ADDREF inside iterator_view_at must hold the parent
   buffer alive until the view is itself unset. */
$big = new NDArray([[1.5, 2.5], [3.5, 4.5]], 'float64');
$row = $big[1];
unset($big);
/* If the parent was released too early, $row would either segfault or
   return garbage. Expect [3.5, 4.5] intact. */
$got = $row->toArray();
echo "outlive: ", ($got == [3.5, 4.5] ? "OK" : "BAD got=" . json_encode($got)), "\n";

/* Two independent sub-views of the same parent must not interfere with
   each other's iterator state. */
$m = new NDArray([[1, 2], [3, 4], [5, 6]], 'int32');
$r0 = $m[0];
$r1 = $m[2];
echo "indep: r0=", json_encode($r0->toArray()), " r2=", json_encode($r1->toArray()), "\n";
?>
--EXPECT--
float32: OK
float64: OK
int32: OK
int64: OK
outlive: OK
indep: r0=[1,2] r2=[5,6]
