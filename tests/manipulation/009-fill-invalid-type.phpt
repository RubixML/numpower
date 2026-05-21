--TEST--
NDArray::fill() rejects non-scalar value types with a clear error
--FILE--
<?php
$a = new NDArray([1, 2, 3], 'float64');

/* PHP array is not a valid fill value — fill broadcasts a single scalar */
try {
    $a->fill([1, 2, 3]);
    echo "BAD: array was accepted\n";
} catch (\Error $e) {
    echo "array: ", $e->getMessage(), "\n";
}

/* Another NDArray is not a valid fill value either */
try {
    $a->fill(new NDArray([1], 'float64'));
    echo "BAD: NDArray was accepted\n";
} catch (\Error $e) {
    echo "object: ", $e->getMessage(), "\n";
}

/* null is not a valid fill value */
try {
    $a->fill(null);
    echo "BAD: null was accepted\n";
} catch (\Error $e) {
    echo "null: ", $e->getMessage(), "\n";
}

/* Verify nothing was written when the call threw — original buffer intact */
echo "intact: ", ($a->toArray() === [1.0, 2.0, 3.0] ? "OK" : "BAD"), "\n";
?>
--EXPECT--
array: Invalid value type. Supported types are: float, int, bool, string
object: Invalid value type. Supported types are: float, int, bool, string
null: Invalid value type. Supported types are: float, int, bool, string
intact: OK
