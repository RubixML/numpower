--TEST--
NDArray::fill() preserves precision for fp128, int64, uint64 boundary values
--FILE--
<?php
/* The scalar fast path uses ndarray_set_from_string for string inputs and
   for IS_LONG when targeting int64/uint64, so it doesn't lose precision
   through a double round-trip. Test the values that previously rounded:
   - fp128 with > 17 mantissa digits
   - int64 with PHP_INT_MAX (overflows float64 mantissa)
   - uint64 with values > PHP_INT_MAX (don't fit a signed long)
*/

/* float128: 34-decimal-digit precision via string */
$a = new NDArray(['0','0','0'], 'float128');
$a->fill('3.14159265358979323846264338327950288');
foreach ($a->toArray() as $i => $v) {
    echo "fp128[$i]=$v\n";
}

/* int64: PHP_INT_MAX as long input must round-trip */
$a = new NDArray([1, 2, 3], 'int64');
$a->fill(PHP_INT_MAX);
$row = $a->toArray();
echo "int64 PHP_INT_MAX: ", ($row === [PHP_INT_MAX, PHP_INT_MAX, PHP_INT_MAX] ? "OK" : "BAD"), "\n";

/* int64: PHP_INT_MIN as long input must round-trip */
$a = new NDArray([1, 2, 3], 'int64');
$a->fill(PHP_INT_MIN);
$row = $a->toArray();
echo "int64 PHP_INT_MIN: ", ($row === [PHP_INT_MIN, PHP_INT_MIN, PHP_INT_MIN] ? "OK" : "BAD"), "\n";

/* uint64: must accept string for values > PHP_INT_MAX */
$a = new NDArray(['0','0','0'], 'uint64');
$a->fill('18446744073709551615');
$row = $a->toArray();
echo "uint64 max: ", ($row === ['18446744073709551615','18446744073709551615','18446744073709551615'] ? "OK" : "BAD"), "\n";

/* float128: IS_LONG of PHP_INT_MAX must keep all 63 bits (the dtype can
   represent it exactly). The old fast path lost ~10 bits going through a
   double cast. */
$a = new NDArray(['0','0','0'], 'float128');
$a->fill(PHP_INT_MAX);
$row = $a->toArray();
echo "float128 PHP_INT_MAX: ", ($row === ['9223372036854775807','9223372036854775807','9223372036854775807'] ? "OK" : "BAD got=" . var_export($row, true)), "\n";

/* float128: IS_LONG of PHP_INT_MIN — same precision requirement, negative. */
$a = new NDArray(['0','0','0'], 'float128');
$a->fill(PHP_INT_MIN);
$row = $a->toArray();
echo "float128 PHP_INT_MIN: ", ($row === ['-9223372036854775808','-9223372036854775808','-9223372036854775808'] ? "OK" : "BAD got=" . var_export($row, true)), "\n";

/* Bool: maps to 1.0 / 0.0 in every dtype's encoding */
$a = new NDArray([1, 2, 3], 'int8');
$a->fill(true);
echo "int8 true: ", ($a->toArray() === [1,1,1] ? "OK" : "BAD"), "\n";

$a = new NDArray([1, 2, 3], 'int8');
$a->fill(false);
echo "int8 false: ", ($a->toArray() === [0,0,0] ? "OK" : "BAD"), "\n";

/* Integer dtype boundary values via long */
$a = new NDArray([0, 0, 0], 'int8');
$a->fill(-128);
echo "int8 -128: ", ($a->toArray() === [-128,-128,-128] ? "OK" : "BAD"), "\n";

$a = new NDArray([0, 0, 0], 'uint16');
$a->fill(65535);
echo "uint16 65535: ", ($a->toArray() === [65535,65535,65535] ? "OK" : "BAD"), "\n";

$a = new NDArray([0, 0, 0], 'int32');
$a->fill(-2147483648);
echo "int32 -2147483648: ", ($a->toArray() === [-2147483648,-2147483648,-2147483648] ? "OK" : "BAD"), "\n";
?>
--EXPECTF--
fp128[0]=3.14159265358979323846264338327950%s
fp128[1]=3.14159265358979323846264338327950%s
fp128[2]=3.14159265358979323846264338327950%s
int64 PHP_INT_MAX: OK
int64 PHP_INT_MIN: OK
uint64 max: OK
float128 PHP_INT_MAX: OK
float128 PHP_INT_MIN: OK
int8 true: OK
int8 false: OK
int8 -128: OK
uint16 65535: OK
int32 -2147483648: OK
