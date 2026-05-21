--TEST--
NDArray index access throws the right exceptions for invalid inputs (CPU)
--FILE--
<?php
/* Element access must throw for:
   - integer index out of range
   - negative index
   - non-integer offset
   For every dtype the same behaviour applies — pick a representative subset
   that crosses the byte-size and string-IO boundaries. The GPU counterpart
   lives in 020-index-access-exceptions-gpu.phpt (skipped when CUDA is off). */

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

    expect_throw("$t CPU index 3 (out of bounds)",
        fn() => (new NDArray($vals, $t))[3]);
    expect_throw("$t CPU index -1 (negative)",
        fn() => (new NDArray($vals, $t))[-1]);
    expect_throw("$t CPU non-int offset",
        fn() => (new NDArray($vals, $t))['foo']);
}
?>
--EXPECT--
float32 CPU index 3 (out of bounds): Index out of bounds
float32 CPU index -1 (negative): Negative indexes are not implemented.
float32 CPU non-int offset: Invalid offset
float128 CPU index 3 (out of bounds): Index out of bounds
float128 CPU index -1 (negative): Negative indexes are not implemented.
float128 CPU non-int offset: Invalid offset
int8 CPU index 3 (out of bounds): Index out of bounds
int8 CPU index -1 (negative): Negative indexes are not implemented.
int8 CPU non-int offset: Invalid offset
int64 CPU index 3 (out of bounds): Index out of bounds
int64 CPU index -1 (negative): Negative indexes are not implemented.
int64 CPU non-int offset: Invalid offset
uint64 CPU index 3 (out of bounds): Index out of bounds
uint64 CPU index -1 (negative): Negative indexes are not implemented.
uint64 CPU non-int offset: Invalid offset
