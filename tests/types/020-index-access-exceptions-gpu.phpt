--TEST--
NDArray index access throws the right exceptions for invalid inputs (GPU)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU counterpart of 020-index-access-exceptions.phpt:
   - integer index out of range on a GPU array
   - negative index on a GPU array
   - toArray() must refuse to run on a GPU array */

$dtypes = ['float32', 'float128', 'int8', 'int64', 'uint64'];

function expect_throw(string $label, callable $f): void {
    try {
        $f();
        echo "$label: NO EXCEPTION\n";
    } catch (\Throwable $e) {
        echo "$label: ", $e->getMessage(), "\n";
    }
}

foreach ($dtypes as $t) {
    $needs_str = in_array($t, ['float128', 'uint64'], true);
    $vals = $needs_str ? ['1', '2', '3'] : [1, 2, 3];

    expect_throw("$t GPU index 3 (out of bounds)",
        fn() => (new NDArray($vals, $t))->gpu()[3]);
    expect_throw("$t GPU index -1 (negative)",
        fn() => (new NDArray($vals, $t))->gpu()[-1]);
    expect_throw("$t toArray() on GPU",
        fn() => (new NDArray($vals, $t))->gpu()->toArray());
}
?>
--EXPECT--
float32 GPU index 3 (out of bounds): Index out of bounds
float32 GPU index -1 (negative): Negative indexes are not implemented.
float32 toArray() on GPU: NDArray must be on CPU RAM before it can be converted to a PHP array.
float128 GPU index 3 (out of bounds): Index out of bounds
float128 GPU index -1 (negative): Negative indexes are not implemented.
float128 toArray() on GPU: NDArray must be on CPU RAM before it can be converted to a PHP array.
int8 GPU index 3 (out of bounds): Index out of bounds
int8 GPU index -1 (negative): Negative indexes are not implemented.
int8 toArray() on GPU: NDArray must be on CPU RAM before it can be converted to a PHP array.
int64 GPU index 3 (out of bounds): Index out of bounds
int64 GPU index -1 (negative): Negative indexes are not implemented.
int64 toArray() on GPU: NDArray must be on CPU RAM before it can be converted to a PHP array.
uint64 GPU index 3 (out of bounds): Index out of bounds
uint64 GPU index -1 (negative): Negative indexes are not implemented.
uint64 toArray() on GPU: NDArray must be on CPU RAM before it can be converted to a PHP array.
