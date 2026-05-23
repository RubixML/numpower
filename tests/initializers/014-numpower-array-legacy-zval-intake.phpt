--TEST--
ZVAL_TO_NDARRAY (raw PHP array in arithmetic) shares modern factory's validation
--FILE--
<?php
/* Arithmetic operations that accept a raw PHP array as one operand route the
 * array through the legacy `Create_NDArray_FromZval` path. Historically this
 * path:
 *   1) bailed on sparse keys (silently NULL);
 *   2) detected only shallow ragged inputs (depth-2+ slipped through and
 *      filled the buffer tail with garbage);
 *   3) raised an uncatchable fatal `E_ERROR` for the inputs it did reject.
 *
 * The legacy entry now delegates to NDArrayFactory_createFromZval so it
 * shares validation with NumPower::array. All three behaviours are tested
 * below. */

function expect_throw(string $label, callable $f): void {
    try {
        $f();
        echo "$label: NO ERROR\n";
    } catch (\Throwable $e) {
        echo "$label: caught Throwable\n";
    }
}

$one_d = NumPower::array([0, 0, 0, 0], 'float32');

/* Sparse 1-D survives the legacy path */
$sparse = [9, 1, 2, 3, 4]; unset($sparse[0]);    /* keys 1..4, values 1..4 */
echo "sparse 1d: ", ($one_d + $sparse), "\n";

/* Shallow ragged (2-D inner-length mismatch) — must throw catchable Error */
expect_throw('shallow_ragged', function () use ($one_d) {
    return $one_d + [[1, 2, 3], [4, 5]];
});

/* Deep ragged (3-D, depth-2 inner mismatch) — must throw, NOT silently fill */
expect_throw('deep_ragged', function () {
    $a = NumPower::array(array_fill(0, 12, 0), 'float32');
    return $a + [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]];
});

/* Mixed scalar/array siblings — must throw */
expect_throw('mixed_sibling', function () use ($one_d) {
    return $one_d + [[1, 2], 3];
});

/* Rectangular 2-D via the legacy path still works */
$two_d = NumPower::array([[0, 0], [0, 0]], 'float32');
echo "rect 2d: ", ($two_d + [[1, 2], [3, 4]]), "\n";
?>
--EXPECT--
sparse 1d: [1, 2, 3, 4]
shallow_ragged: caught Throwable
deep_ragged: caught Throwable
mixed_sibling: caught Throwable
rect 2d: [[1, 2]
 [3, 4]]
