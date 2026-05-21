--TEST--
NDArray::fill() preserves boundary values exactly for each integer dtype, and special floats
--FILE--
<?php
/* Per CLAUDE.md the test should hit boundary values for every dtype so we'd
   catch a "off by one bit" regression in the encode path. Each row writes
   the named boundary value and reads it back via toArray(). */

$cases = [
    /* dtype     boundary value          expected readback */
    ['int8',    -128,                   -128],
    ['int8',     127,                    127],
    ['uint8',    0,                      0],
    ['uint8',    255,                    255],
    ['int16',   -32768,                  -32768],
    ['int16',    32767,                  32767],
    ['uint16',   0,                      0],
    ['uint16',   65535,                  65535],
    ['int32',   -2147483648,            -2147483648],
    ['int32',    2147483647,             2147483647],
    ['uint32',   0,                      0],
    ['uint32',   4294967295,             4294967295],
    ['int64',    PHP_INT_MIN,            PHP_INT_MIN],
    ['int64',    PHP_INT_MAX,            PHP_INT_MAX],
];

foreach ($cases as [$t, $v, $exp]) {
    $a = new NDArray([0, 0, 0], $t);
    $a->fill($v);
    $row = $a->toArray();
    $ok = ($row === [$exp, $exp, $exp]);
    echo "$t fill($v): ", $ok ? "OK" : "BAD got=" . var_export($row, true), "\n";
}

/* uint64 max via string (doesn't fit signed long) */
$a = new NDArray(['0','0','0'], 'uint64');
$a->fill('18446744073709551615');
$row = $a->toArray();
echo "uint64 fill('18446744073709551615'): ",
     ($row === ['18446744073709551615','18446744073709551615','18446744073709551615'] ? "OK" : "BAD"), "\n";

/* Special floats — INF / -INF / NaN must round-trip on float32/float64/float128 */
foreach (['float32', 'float64', 'float128'] as $t) {
    $a = new NDArray(['0','0','0'], $t);
    $a->fill(INF);
    $row = $a->toArray();
    /* float128 stringifies as 'inf' (string); float32/64 stringify as INF (double) */
    $first = $row[0];
    $allInf = ($first === $row[1] && $first === $row[2]);
    $isInf  = (is_float($first) && $first === INF) || (is_string($first) && (string)$first === 'inf');
    echo "$t fill(INF): ", ($allInf && $isInf ? "OK" : "BAD got=" . var_export($row, true)), "\n";

    $a = new NDArray(['0','0','0'], $t);
    $a->fill(-INF);
    $row = $a->toArray();
    $first = $row[0];
    $allNegInf = ($first === $row[1] && $first === $row[2]);
    $isNegInf  = (is_float($first) && $first === -INF) || (is_string($first) && (string)$first === '-inf');
    echo "$t fill(-INF): ", ($allNegInf && $isNegInf ? "OK" : "BAD got=" . var_export($row, true)), "\n";

    $a = new NDArray(['0','0','0'], $t);
    $a->fill(NAN);
    $row = $a->toArray();
    /* NAN != NAN — compare each element with is_nan */
    $allNan = true;
    foreach ($row as $v) {
        $isNan = (is_float($v) && is_nan($v)) || (is_string($v) && (string)$v === 'nan');
        if (!$isNan) { $allNan = false; break; }
    }
    echo "$t fill(NAN): ", ($allNan ? "OK" : "BAD got=" . var_export($row, true)), "\n";
}

/* Garbage strings parse as 0 (strtod / strtoll convention) — document that
   behavior so callers know not to rely on validation. */
$a = new NDArray([1, 2, 3], 'float64');
$a->fill('not-a-number');
echo "float64 fill('not-a-number'): ", ($a->toArray() === [0.0, 0.0, 0.0] ? "OK" : "BAD"), "\n";

$a = new NDArray([1, 2, 3], 'int32');
$a->fill('also-garbage');
echo "int32 fill('also-garbage'): ", ($a->toArray() === [0, 0, 0] ? "OK" : "BAD"), "\n";

/* Empty string also parses as 0 */
$a = new NDArray([1, 2, 3], 'float64');
$a->fill('');
echo "float64 fill(''): ", ($a->toArray() === [0.0, 0.0, 0.0] ? "OK" : "BAD"), "\n";
?>
--EXPECT--
int8 fill(-128): OK
int8 fill(127): OK
uint8 fill(0): OK
uint8 fill(255): OK
int16 fill(-32768): OK
int16 fill(32767): OK
uint16 fill(0): OK
uint16 fill(65535): OK
int32 fill(-2147483648): OK
int32 fill(2147483647): OK
uint32 fill(0): OK
uint32 fill(4294967295): OK
int64 fill(-9223372036854775808): OK
int64 fill(9223372036854775807): OK
uint64 fill('18446744073709551615'): OK
float32 fill(INF): OK
float32 fill(-INF): OK
float32 fill(NAN): OK
float64 fill(INF): OK
float64 fill(-INF): OK
float64 fill(NAN): OK
float128 fill(INF): OK
float128 fill(-INF): OK
float128 fill(NAN): OK
float64 fill('not-a-number'): OK
int32 fill('also-garbage'): OK
float64 fill(''): OK
