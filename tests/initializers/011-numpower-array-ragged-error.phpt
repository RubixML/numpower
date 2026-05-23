--TEST--
NumPower::array rejects non-rectangular (ragged) / mixed PHP arrays with a clear error
--FILE--
<?php
/* PyTorch / modern NumPy raise on ragged input; we match that behaviour so
 * a ragged input is not silently filled into a rectangular buffer with
 * uninitialised tail slots (the prior behaviour). */

function expect_error(string $label, callable $fn): void {
    try {
        $arr = $fn();
        echo "$label: NO ERROR — got shape ", json_encode($arr->shape()), "\n";
    } catch (\Throwable $e) {
        echo "$label: ", $e->getMessage(), "\n";
    }
}

/* Different inner lengths — classic ragged. */
expect_error('ragged_2d',          fn() => NumPower::array([[1, 2, 3], [4, 5]], 'float32'));
expect_error('ragged_2d_long_2nd', fn() => NumPower::array([[1, 2], [3, 4, 5]], 'float32'));
expect_error('ragged_2d_dtype',    fn() => NumPower::array([[1, 2, 3], [4, 5]], 'float128'));

/* Sparse outer that *becomes* ragged after the unset. */
expect_error('sparse_then_ragged', function () {
    $n = [[1, 2, 3], [4, 5, 6]];
    unset($n[0][0]);  /* row 0 now has 2 elements, row 1 still has 3 */
    return NumPower::array($n, 'float64');
});

/* A scalar sibling next to an array sibling at the same depth. */
expect_error('mixed_sibling_scalar_first', fn() => NumPower::array([1, [2, 3]], 'float32'));
expect_error('mixed_sibling_array_first',  fn() => NumPower::array([[1, 2], 3], 'float32'));

/* Empty arrays — both fully empty and "list of empty lists" — are NOT ragged. */
$empty1d = NumPower::array([], 'float32');
echo "empty_1d shape: ", json_encode($empty1d->shape()), "\n";

$emptyrows = NumPower::array([[], [], []], 'float32');
echo "empty_rows shape: ", json_encode($emptyrows->shape()), "\n";

/* Sanity: rectangular sparse is still accepted (regression guard). */
$m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
unset($m[0]);                 /* outer keys: [1, 2] — sparse but rectangular */
$arr = NumPower::array($m, 'int32');
echo "sparse_rectangular: ", $arr, "\n";
?>
--EXPECT--
ragged_2d: Cannot build NDArray from a non-rectangular (ragged) array: expected 3 elements, got 2.
ragged_2d_long_2nd: Cannot build NDArray from a non-rectangular (ragged) array: expected 2 elements, got 3.
ragged_2d_dtype: Cannot build NDArray from a non-rectangular (ragged) array: expected 3 elements, got 2.
sparse_then_ragged: Cannot build NDArray from a non-rectangular (ragged) array: expected 2 elements, got 3.
mixed_sibling_scalar_first: Cannot build NDArray: mixed scalar/array siblings at the same depth.
mixed_sibling_array_first: Cannot build NDArray: mixed scalar/array siblings at the same depth.
empty_1d shape: [0]
empty_rows shape: [3,0]
sparse_rectangular: [[4, 5, 6]
 [7, 8, 9]]
