--TEST--
NDArray::fill() broadcasts a scalar across every element, dtype-correct on CPU
--FILE--
<?php
/* fill() now accepts string in addition to int/float/bool. The written value
   is encoded in the target's dtype, so reading back via toArray() returns
   the dtype-mandated PHP type (string for fp128/uint64, int for int8..int64,
   float otherwise). The source-of-truth for "what does this scalar look like
   in dtype T" is the constructor: NDArray([v], T)[0] does the same encoding.

   Values chosen to be exactly representable in every dtype (3, 0, 1) so the
   test doesn't depend on fp4/fp8 quantisation rules. */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

function fmt($v) { return is_string($v) ? "'" . $v . "'" : var_export($v, true); }

foreach ($types as $t) {
    /* String fill */
    $a = new NDArray([1, 2, 3, 4], $t);
    $a->fill('3');
    $expected = (new NDArray(['3'], $t))[0];
    $row = $a->toArray();
    $ok = ($row === [$expected, $expected, $expected, $expected]);
    echo "$t fill('3'): ", $ok ? "OK" : "BAD got=" . var_export($row, true), "\n";

    /* Int fill */
    $a = new NDArray([1, 2, 3, 4], $t);
    $a->fill(1);
    $expected = (new NDArray(['1'], $t))[0];
    $ok = ($a->toArray() === [$expected, $expected, $expected, $expected]);
    echo "$t fill(1): ", $ok ? "OK" : "BAD", "\n";

    /* Float fill */
    $a = new NDArray([1, 2, 3, 4], $t);
    $a->fill(2.0);
    $expected = (new NDArray(['2'], $t))[0];
    $ok = ($a->toArray() === [$expected, $expected, $expected, $expected]);
    echo "$t fill(2.0): ", $ok ? "OK" : "BAD", "\n";

    /* Bool true → 1, false → 0 */
    $a = new NDArray([1, 2, 3, 4], $t);
    $a->fill(true);
    $expected_true = (new NDArray(['1'], $t))[0];
    $ok = ($a->toArray() === [$expected_true, $expected_true, $expected_true, $expected_true]);
    echo "$t fill(true): ", $ok ? "OK" : "BAD", "\n";

    $a = new NDArray([1, 2, 3, 4], $t);
    $a->fill(false);
    $expected_false = (new NDArray(['0'], $t))[0];
    $ok = ($a->toArray() === [$expected_false, $expected_false, $expected_false, $expected_false]);
    echo "$t fill(false): ", $ok ? "OK" : "BAD", "\n";
}

/* 2-D fill — every cell of every row must hold the broadcast value */
$a = new NDArray([[1,2,3],[4,5,6]], 'float64');
$a->fill('2.5');
echo "float64 2D fill('2.5'): ",
     ($a->toArray() === [[2.5,2.5,2.5],[2.5,2.5,2.5]] ? "OK" : "BAD"), "\n";

/* 3-D fill */
$a = new NDArray([[[1,2],[3,4]],[[5,6],[7,8]]], 'int32');
$a->fill(9);
echo "int32 3D fill(9): ",
     ($a->toArray() === [[[9,9],[9,9]],[[9,9],[9,9]]] ? "OK" : "BAD"), "\n";
?>
--EXPECT--
float4 fill('3'): OK
float4 fill(1): OK
float4 fill(2.0): OK
float4 fill(true): OK
float4 fill(false): OK
float8 fill('3'): OK
float8 fill(1): OK
float8 fill(2.0): OK
float8 fill(true): OK
float8 fill(false): OK
float16 fill('3'): OK
float16 fill(1): OK
float16 fill(2.0): OK
float16 fill(true): OK
float16 fill(false): OK
float32 fill('3'): OK
float32 fill(1): OK
float32 fill(2.0): OK
float32 fill(true): OK
float32 fill(false): OK
float64 fill('3'): OK
float64 fill(1): OK
float64 fill(2.0): OK
float64 fill(true): OK
float64 fill(false): OK
float128 fill('3'): OK
float128 fill(1): OK
float128 fill(2.0): OK
float128 fill(true): OK
float128 fill(false): OK
int8 fill('3'): OK
int8 fill(1): OK
int8 fill(2.0): OK
int8 fill(true): OK
int8 fill(false): OK
uint8 fill('3'): OK
uint8 fill(1): OK
uint8 fill(2.0): OK
uint8 fill(true): OK
uint8 fill(false): OK
int16 fill('3'): OK
int16 fill(1): OK
int16 fill(2.0): OK
int16 fill(true): OK
int16 fill(false): OK
uint16 fill('3'): OK
uint16 fill(1): OK
uint16 fill(2.0): OK
uint16 fill(true): OK
uint16 fill(false): OK
int32 fill('3'): OK
int32 fill(1): OK
int32 fill(2.0): OK
int32 fill(true): OK
int32 fill(false): OK
uint32 fill('3'): OK
uint32 fill(1): OK
uint32 fill(2.0): OK
uint32 fill(true): OK
uint32 fill(false): OK
int64 fill('3'): OK
int64 fill(1): OK
int64 fill(2.0): OK
int64 fill(true): OK
int64 fill(false): OK
uint64 fill('3'): OK
uint64 fill(1): OK
uint64 fill(2.0): OK
uint64 fill(true): OK
uint64 fill(false): OK
float64 2D fill('2.5'): OK
int32 3D fill(9): OK
