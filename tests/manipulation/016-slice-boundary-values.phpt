--TEST--
NumPower::slice() preserves boundary values exactly for integer dtypes and special floats
--FILE--
<?php
/* slice() must copy bytes — not reinterpret them — so the boundary values
   that round-trip through the constructor must also round-trip through a
   slice. This is what was broken by the original code's hardcoded float32
   descriptor: any non-float32 dtype was reinterpreted on slice. */

$int_boundaries = [
    'int8'   => ['-128', '127'],
    'uint8'  => ['0', '255'],
    'int16'  => ['-32768', '32767'],
    'uint16' => ['0', '65535'],
    'int32'  => ['-2147483648', '2147483647'],
    'uint32' => ['0', '4294967295'],
    'int64'  => ['-9223372036854775808', '9223372036854775807'],
    'uint64' => ['0', '18446744073709551615'],
];

foreach ($int_boundaries as $t => [$lo, $hi]) {
    /* 1-D slice keeps both ends through a [start,stop] copy */
    $v = new NDArray([$lo, '0', $hi], $t);
    $s = NumPower::slice($v, [0, 3]);
    $expected = [
        (new NDArray([$lo], $t))[0],
        (new NDArray(['0'], $t))[0],
        (new NDArray([$hi], $t))[0],
    ];
    $ok = ($s->toArray() === $expected);
    echo "$t boundary 1D [0,3): ", $ok ? "OK" : "BAD got=" . var_export($s->toArray(), true), "\n";

    /* Single-int reduction must return exact scalar */
    $hi_scalar = NumPower::slice($v, 2);
    $expected_hi = (new NDArray([$hi], $t))[0];
    $ok = ($hi_scalar === $expected_hi);
    echo "$t slice(hi): ", $ok ? "OK" : "BAD got=" . var_export($hi_scalar, true), "\n";

    $lo_scalar = NumPower::slice($v, 0);
    $expected_lo = (new NDArray([$lo], $t))[0];
    $ok = ($lo_scalar === $expected_lo);
    echo "$t slice(lo): ", $ok ? "OK" : "BAD got=" . var_export($lo_scalar, true), "\n";

    /* Stride pattern that copies one of each boundary */
    $w = new NDArray([$lo, '1', $hi, '2'], $t);
    $stepped = NumPower::slice($w, [0, 4, 2]);
    $expected = [
        (new NDArray([$lo], $t))[0],
        (new NDArray([$hi], $t))[0],
    ];
    $ok = ($stepped->toArray() === $expected);
    echo "$t stride boundaries: ", $ok ? "OK" : "BAD", "\n";
}

/* Special floats: INF, -INF, NaN, +0, -0 across float dtypes (where representable).
   Detect signed-zero through the raw bit pattern (PHP 8 throws DivisionByZeroError
   for `1.0/-0.0`, so we can't use that test). */
function is_negzero(float $v): bool {
    return $v === 0.0 && ord(pack('E', $v)[0]) >= 0x80;
}

$float_special = ['float32', 'float64'];
foreach ($float_special as $t) {
    $vals = [INF, -INF, NAN, 0.0, -0.0];
    $v = new NDArray($vals, $t);
    $s = NumPower::slice($v, [0, 5]);
    $row = $s->toArray();
    $ok =  $row[0] === INF
        && $row[1] === -INF
        && is_nan($row[2])
        && $row[3] === 0.0
        && !is_negzero($row[3])
        && $row[4] === 0.0
        && is_negzero($row[4]);
    echo "$t special floats: ", $ok ? "OK" : "BAD got=" . var_export($row, true), "\n";

    /* Single-int gives scalar with same sign */
    $neg0 = NumPower::slice($v, 4);
    $ok = (is_float($neg0) && is_negzero($neg0));
    echo "$t slice(-0) sign: ", $ok ? "OK" : "BAD got=" . var_export($neg0, true), "\n";
}

/* fp128 boundary: a value beyond fp64 range must survive slicing. */
$fp128_big = '1.2345678901234567890123456789012345e300';
$v = new NDArray([$fp128_big, '0', $fp128_big], 'float128');
$s = NumPower::slice($v, [0, 3, 2]);
$expected = [
    (new NDArray([$fp128_big], 'float128'))[0],
    (new NDArray([$fp128_big], 'float128'))[0],
];
$ok = ($s->toArray() === $expected);
echo "float128 huge stride: ", $ok ? "OK" : "BAD", "\n";
?>
--EXPECT--
int8 boundary 1D [0,3): OK
int8 slice(hi): OK
int8 slice(lo): OK
int8 stride boundaries: OK
uint8 boundary 1D [0,3): OK
uint8 slice(hi): OK
uint8 slice(lo): OK
uint8 stride boundaries: OK
int16 boundary 1D [0,3): OK
int16 slice(hi): OK
int16 slice(lo): OK
int16 stride boundaries: OK
uint16 boundary 1D [0,3): OK
uint16 slice(hi): OK
uint16 slice(lo): OK
uint16 stride boundaries: OK
int32 boundary 1D [0,3): OK
int32 slice(hi): OK
int32 slice(lo): OK
int32 stride boundaries: OK
uint32 boundary 1D [0,3): OK
uint32 slice(hi): OK
uint32 slice(lo): OK
uint32 stride boundaries: OK
int64 boundary 1D [0,3): OK
int64 slice(hi): OK
int64 slice(lo): OK
int64 stride boundaries: OK
uint64 boundary 1D [0,3): OK
uint64 slice(hi): OK
uint64 slice(lo): OK
uint64 stride boundaries: OK
float32 special floats: OK
float32 slice(-0) sign: OK
float64 special floats: OK
float64 slice(-0) sign: OK
float128 huge stride: OK
