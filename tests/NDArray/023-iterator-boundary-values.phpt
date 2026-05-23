--TEST--
NDArray Iterator: boundary values come back from current() with full precision
--FILE--
<?php
/* current() routes 1-D results through NDArray_ScalarToZval, the same dtype-
   aware path as $a[i]. We verify that path preserves boundary values for the
   dtypes whose native PHP type is something other than float -- that is, the
   ones where the obvious "double round-trip" would lose precision:
     - int64  -> PHP int, must reach PHP_INT_MIN/MAX exactly
     - uint64 -> PHP string, must reach 18446744073709551615 exactly
     - float128 -> PHP string, must reach a >17-digit mantissa losslessly
   Single-byte ints (int8/uint8) and their 16/32 counterparts are also
   checked at their extremes to confirm the sign bit survives. */

/* int8: min/max. */
$a = new NDArray([-128, 127, 0, -1], 'int8');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "int8: ", implode(',', $seen), "\n";

/* uint8: top of range. */
$a = new NDArray([0, 1, 254, 255], 'uint8');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "uint8: ", implode(',', $seen), "\n";

/* int16, uint16. */
$a = new NDArray([-32768, 32767, 0], 'int16');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "int16: ", implode(',', $seen), "\n";

$a = new NDArray([0, 65535], 'uint16');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "uint16: ", implode(',', $seen), "\n";

/* int32, uint32. */
$a = new NDArray([-2147483648, 2147483647, 0], 'int32');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "int32: ", implode(',', $seen), "\n";

$a = new NDArray([0, 4294967295], 'uint32');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "uint32: ", implode(',', $seen), "\n";

/* int64: PHP_INT_MIN / PHP_INT_MAX -- these are the boundaries where a
   detour through float32/float64 would lose precision. */
$a = new NDArray(['-9223372036854775808', '9223372036854775807', '0'], 'int64');
$seen = [];
foreach ($a as $v) { $seen[] = is_int($v) ? (string)$v : '!notint'; }
echo "int64: ", implode(',', $seen), "\n";

/* uint64: returned as decimal string to carry values > PHP_INT_MAX. */
$a = new NDArray(['0', '18446744073709551615', '9223372036854775808'], 'uint64');
$seen = [];
foreach ($a as $v) { $seen[] = is_string($v) ? $v : '!notstr'; }
echo "uint64: ", implode(',', $seen), "\n";

/* float128: decimal-string output preserves more than ~17 digits. */
$a = new NDArray(['1.2345678901234567890123456789', '0', '-0.5'], 'float128');
$seen = [];
foreach ($a as $v) { $seen[] = is_string($v) ? $v : '!notstr'; }
echo "float128 types: ", implode('|', array_map('gettype', iterator_to_array($a, false))), "\n";
/* We only assert "more than double-precision" rather than an exact string,
   because dd vs libquadmath formats differ across platforms. */
$dd_or_quadmath = preg_match('/^1\.23456789012345678/', $seen[0]);
echo "float128 first val starts with >17 digits of mantissa: ",
     $dd_or_quadmath ? '1' : '0', "\n";
echo "float128 size=", count($seen), "\n";

/* float types: simple round-trip sanity. */
$a = new NDArray([1.5, -2.5, 0.0, 1e10], 'float64');
$seen = [];
foreach ($a as $v) { $seen[] = is_float($v) ? (string)$v : '!notf'; }
echo "float64: ", implode(',', $seen), "\n";
?>
--EXPECT--
int8: -128,127,0,-1
uint8: 0,1,254,255
int16: -32768,32767,0
uint16: 0,65535
int32: -2147483648,2147483647,0
uint32: 0,4294967295
int64: -9223372036854775808,9223372036854775807,0
uint64: 0,18446744073709551615,9223372036854775808
float128 types: string|string|string
float128 first val starts with >17 digits of mantissa: 1
float128 size=3
float64: 1.5,-2.5,0,10000000000
