--TEST--
NDArray::slice() preserves dtype for every supported dtype on CPU
--FILE--
<?php
/* slice() must NOT silently convert the source dtype to float32. The result's
   element type must match the source. Boundary values are chosen to be
   exactly representable in the integer ranges and to fall safely within fp4/fp8
   quantisation. For fp128/uint64 we use NDArray([v], T)[0] as the source-of-
   truth for "what does this scalar look like in dtype T".

   Edge cases covered per dtype:
   - 1-D contiguous slice (no step)
   - 1-D dim reduction via single int (returns scalar)
   - 2-D slice([], -1) (column extraction — non-contiguous source memory pattern)
   - 2-D slice([0, 2]) (row range)
   - 3-D slice(1) (drop leading axis)
   - Empty slice ([3, 3] → shape [0]) preserves dtype on the empty result */

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

function ensure_dtype(NDArray $a, string $expected, string $label) {
    /* No public dtype getter, but the toArray() PHP type is dtype-determined. */
    $row = $a->toArray();
    while (is_array($row)) $row = $row[0] ?? null;
    $exp_php_type = match (true) {
        in_array($expected, ['float128','uint64'], true) => 'string',
        in_array($expected, ['int8','uint8','int16','uint16','int32','uint32','int64'], true) => 'integer',
        default => 'double',
    };
    $got = gettype($row);
    if ($got !== $exp_php_type) {
        echo "$label dtype=$expected: BAD got PHP type=$got expected=$exp_php_type\n";
    } else {
        echo "$label dtype=$expected: OK\n";
    }
}

foreach ($types as $t) {
    /* 1-D slice [start, stop] preserves dtype + values. Using the static
       non-mutating form keeps the dtype-specific source around for the
       follow-up sub-tests. */
    $v = new NDArray([1, 2, 3, 4, 5, 6], $t);
    $s = NumPower::slice($v, [1, 4]);
    $expected_vals = [
        (new NDArray(['2'], $t))[0],
        (new NDArray(['3'], $t))[0],
        (new NDArray(['4'], $t))[0],
    ];
    $ok = ($s->toArray() === $expected_vals);
    echo "$t 1D [1,4): ", $ok ? "OK" : "BAD got=" . var_export($s->toArray(), true), "\n";
    ensure_dtype($s, $t, "$t 1D [1,4)");

    /* Single-int slice on 1-D returns scalar of dtype-correct PHP type */
    $v2 = new NDArray([1, 2, 3], $t);
    $scalar = NumPower::slice($v2, 1);
    $expected_scalar = (new NDArray(['2'], $t))[0];
    $ok = ($scalar === $expected_scalar);
    echo "$t 1D slice(1) scalar: ", $ok ? "OK" : "BAD got=" . var_export($scalar, true), "\n";

    /* 2-D last column (slice([], -1)) — non-contiguous source-memory pattern */
    $m = new NDArray([[1, 2, 3], [4, 5, 6]], $t);
    $col = NumPower::slice($m, [], -1);
    $expected_col = [
        (new NDArray(['3'], $t))[0],
        (new NDArray(['6'], $t))[0],
    ];
    $ok = ($col->toArray() === $expected_col);
    echo "$t 2D last col: ", $ok ? "OK" : "BAD got=" . var_export($col->toArray(), true), "\n";

    /* 2-D row slice */
    $row = NumPower::slice($m, 0);
    $expected_row = [
        (new NDArray(['1'], $t))[0],
        (new NDArray(['2'], $t))[0],
        (new NDArray(['3'], $t))[0],
    ];
    $ok = ($row->toArray() === $expected_row);
    echo "$t 2D row 0: ", $ok ? "OK" : "BAD got=" . var_export($row->toArray(), true), "\n";

    /* Empty slice keeps dtype + has shape [0] */
    $empty = NumPower::slice(new NDArray([1, 2, 3], $t), [2, 2]);
    $ok = ($empty->shape() === [0] && $empty->toArray() === []);
    echo "$t empty slice: ", $ok ? "OK" : "BAD shape=" . var_export($empty->shape(), true), "\n";
}

/* 3-D slice with dim reduction across each axis position. */
$cube = new NDArray([[[1,2],[3,4]],[[5,6],[7,8]]], 'int32');
$ok0 = (NumPower::slice($cube, 1)->toArray() === [[5,6],[7,8]]);
$ok1 = (NumPower::slice($cube, [], 1)->toArray() === [[3,4],[7,8]]);
$ok2 = (NumPower::slice($cube, [], [], 1)->toArray() === [[2,4],[6,8]]);
echo "int32 3D drop axis 0: ", $ok0 ? "OK" : "BAD", "\n";
echo "int32 3D drop axis 1: ", $ok1 ? "OK" : "BAD", "\n";
echo "int32 3D drop axis 2: ", $ok2 ? "OK" : "BAD", "\n";
?>
--EXPECT--
float4 1D [1,4): OK
float4 1D [1,4) dtype=float4: OK
float4 1D slice(1) scalar: OK
float4 2D last col: OK
float4 2D row 0: OK
float4 empty slice: OK
float8 1D [1,4): OK
float8 1D [1,4) dtype=float8: OK
float8 1D slice(1) scalar: OK
float8 2D last col: OK
float8 2D row 0: OK
float8 empty slice: OK
float16 1D [1,4): OK
float16 1D [1,4) dtype=float16: OK
float16 1D slice(1) scalar: OK
float16 2D last col: OK
float16 2D row 0: OK
float16 empty slice: OK
float32 1D [1,4): OK
float32 1D [1,4) dtype=float32: OK
float32 1D slice(1) scalar: OK
float32 2D last col: OK
float32 2D row 0: OK
float32 empty slice: OK
float64 1D [1,4): OK
float64 1D [1,4) dtype=float64: OK
float64 1D slice(1) scalar: OK
float64 2D last col: OK
float64 2D row 0: OK
float64 empty slice: OK
float128 1D [1,4): OK
float128 1D [1,4) dtype=float128: OK
float128 1D slice(1) scalar: OK
float128 2D last col: OK
float128 2D row 0: OK
float128 empty slice: OK
int8 1D [1,4): OK
int8 1D [1,4) dtype=int8: OK
int8 1D slice(1) scalar: OK
int8 2D last col: OK
int8 2D row 0: OK
int8 empty slice: OK
uint8 1D [1,4): OK
uint8 1D [1,4) dtype=uint8: OK
uint8 1D slice(1) scalar: OK
uint8 2D last col: OK
uint8 2D row 0: OK
uint8 empty slice: OK
int16 1D [1,4): OK
int16 1D [1,4) dtype=int16: OK
int16 1D slice(1) scalar: OK
int16 2D last col: OK
int16 2D row 0: OK
int16 empty slice: OK
uint16 1D [1,4): OK
uint16 1D [1,4) dtype=uint16: OK
uint16 1D slice(1) scalar: OK
uint16 2D last col: OK
uint16 2D row 0: OK
uint16 empty slice: OK
int32 1D [1,4): OK
int32 1D [1,4) dtype=int32: OK
int32 1D slice(1) scalar: OK
int32 2D last col: OK
int32 2D row 0: OK
int32 empty slice: OK
uint32 1D [1,4): OK
uint32 1D [1,4) dtype=uint32: OK
uint32 1D slice(1) scalar: OK
uint32 2D last col: OK
uint32 2D row 0: OK
uint32 empty slice: OK
int64 1D [1,4): OK
int64 1D [1,4) dtype=int64: OK
int64 1D slice(1) scalar: OK
int64 2D last col: OK
int64 2D row 0: OK
int64 empty slice: OK
uint64 1D [1,4): OK
uint64 1D [1,4) dtype=uint64: OK
uint64 1D slice(1) scalar: OK
uint64 2D last col: OK
uint64 2D row 0: OK
uint64 empty slice: OK
int32 3D drop axis 0: OK
int32 3D drop axis 1: OK
int32 3D drop axis 2: OK
