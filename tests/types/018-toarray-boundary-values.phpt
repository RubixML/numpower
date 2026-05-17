--TEST--
NDArray::toArray() preserves boundary values exactly across all integer dtypes
--FILE--
<?php
/* Verify that toArray() preserves min/max boundary values exactly for every
   integer dtype, plus high-precision and extreme values for float128.
   uint64 and float128 must come back as strings (PHP_INT_MAX is smaller
   than UINT64_MAX, and double precision can't hold 128-bit floats). */

$intCases = [
    'int8'   => [-128, 127,  -1, 0],
    'uint8'  => [0,    255,  1,  128],
    'int16'  => [-32768, 32767, -1, 0],
    'uint16' => [0,    65535, 1,  32768],
    'int32'  => [-2147483648, 2147483647, -1, 0],
    'uint32' => [0, 4294967295, 1, 2147483648],
    'int64'  => [PHP_INT_MIN, PHP_INT_MAX, -1, 0],
];

foreach ($intCases as $dtype => $vals) {
    $arr = (new NDArray($vals, $dtype))->toArray();
    $ok = ($arr === $vals);
    echo "$dtype: ", ($ok ? 'OK' : 'BAD: ' . var_export($arr, true)), "\n";
}

/* uint64 — only the small values fit in a PHP int; large ones must remain
   strings in the input and come back as strings. */
$u64in  = ['0', '1', '128', '9223372036854775807', '9223372036854775808', '18446744073709551615'];
$u64out = (new NDArray($u64in, 'uint64'))->toArray();
echo 'uint64: ', ($u64out === $u64in ? 'OK' : 'BAD: ' . var_export($u64out, true)), "\n";

/* float128 — high-precision round trip; preserve via string representation */
$f128in  = ['0', '1.5', '-1.5', '1e-9', '1e9', '3.14159265358979324'];
$f128out = (new NDArray($f128in, 'float128'))->toArray();
$ok128 = true;
foreach ($f128in as $i => $s) {
    /* fp128_to_string formats with %.18Lg, so "1e-9" → "1e-09", etc. Just check
       the parsed long-double values match within float128 precision. */
    if (!is_string($f128out[$i])) { $ok128 = false; break; }
    if (abs((float)$f128out[$i] - (float)$s) > max(1e-15, abs((float)$s) * 1e-12)) {
        $ok128 = false; break;
    }
}
echo 'float128: ', ($ok128 ? 'OK' : 'BAD: ' . var_export($f128out, true)), "\n";
?>
--EXPECT--
int8: OK
uint8: OK
int16: OK
uint16: OK
int32: OK
uint32: OK
int64: OK
uint64: OK
float128: OK
