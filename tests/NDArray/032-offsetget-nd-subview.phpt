--TEST--
NDArray::offsetGet() on N-D source returns an (N-1)-D sub-view with the source dtype
--FILE--
<?php
/* For ndim >= 2 source, offsetGet returns an NDArray sub-view that:
     - has rank N-1
     - preserves the source dtype
     - has shape == source.shape[1:]
     - aliases the source buffer (chained $a[i][j] reads work, no copy)

   This covers the path through iterator_view_at + ndarray_init_new_object
   for the non-scalar case. */

$cases = [
    'float32'  => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    'float64'  => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    'float128' => [['1', '2', '3'], ['4', '5', '6']],
    'int8'     => [[-128, 0, 127], [-1, 1, 64]],
    'uint8'    => [[0, 128, 255], [1, 64, 200]],
    'int32'    => [[-2147483648, 0, 2147483647], [-1, 7, 42]],
    'int64'    => [[PHP_INT_MIN, 0, PHP_INT_MAX], [-1, 7, 42]],
    'uint64'   => [['0', '1', '18446744073709551615'], ['7', '42', '100']],
];

foreach ($cases as $dtype => $rows) {
    $a = new NDArray($rows, $dtype);
    $row0 = $a[0];
    $row1 = $a[1];

    /* Sub-view is an NDArray of rank N-1. */
    $is_obj   = $row0 instanceof NDArray;
    $shape_ok = $row0->shape() === [3] && $row1->shape() === [3];

    /* Chained scalar access through the sub-view returns the right typed scalar. */
    $first_match = (string)$row0[0] === (string)$rows[0][0];
    $last_match  = (string)$row1[2] === (string)$rows[1][2];

    echo "$dtype: obj=", $is_obj ? "OK" : "BAD",
         " shape=", $shape_ok ? "OK" : "BAD",
         " first=", $first_match ? "OK" : "BAD",
         " last=", $last_match ? "OK" : "BAD", "\n";
}

/* 3-D source: $a[i] -> 2-D, $a[i][j] -> 1-D, $a[i][j][k] -> dtype-correct scalar. */
$cube = new NDArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 'int32');
$plane = $cube[1];
$line  = $cube[1][0];
echo "3-D: plane=", $plane->shape() === [2, 2] ? "OK" : "BAD",
     " line=", $line->shape() === [2] ? "OK" : "BAD",
     " scalar=", $cube[1][1][1] === 8 ? "OK" : "BAD",
     " scalar_type=", is_int($cube[1][1][1]) ? "OK" : "BAD", "\n";
?>
--EXPECT--
float32: obj=OK shape=OK first=OK last=OK
float64: obj=OK shape=OK first=OK last=OK
float128: obj=OK shape=OK first=OK last=OK
int8: obj=OK shape=OK first=OK last=OK
uint8: obj=OK shape=OK first=OK last=OK
int32: obj=OK shape=OK first=OK last=OK
int64: obj=OK shape=OK first=OK last=OK
uint64: obj=OK shape=OK first=OK last=OK
3-D: plane=OK line=OK scalar=OK scalar_type=OK
