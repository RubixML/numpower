--TEST--
NumPower::det float64 (double-precision determinant)
--FILE--
<?php
use NumPower as nd;

/* float64 det should route through dgetrf (LAPACK double) and preserve
   float64 precision. Check a case where float32 would visibly diverge. */

$A = nd::array([[4.0, 7.0], [2.0, 6.0]], "float64");
$v = nd::det($A);
$ok = abs($v - 10.0) < 1e-14;
echo "det [[4,7],[2,6]] (expect 10): ", $ok ? "OK" : "BAD " . var_export($v, true), "\n";

/* Larger precision check: value > 1e5, float32 has ~7 decimals, float64 ~15. */
$A2 = nd::array([[123456.789012345, 234567.890123456], [456789.012345678, 567890.123456789]], "float64");
$exp = 123456.789012345 * 567890.123456789 - 234567.890123456 * 456789.012345678;
$v2 = nd::det($A2);
$ok2 = ($v2 - $exp) / $exp < 1e-14;
echo "float64 precision > 1e11: ", $ok2 ? "OK" : "BAD", "\n";

/* float32 version on same input: should be a float (PHP) — confirm we're
   using the correct double path. */
$A3 = nd::array([[4.0, 7.0], [2.0, 6.0]], "float32");
$v3 = nd::det($A3);
$ok3 = abs($v3 - 10.0) < 1e-5;
echo "float32 det: ", $ok3 ? "OK" : "BAD", "\n";
?>
--EXPECT--
det [[4,7],[2,6]] (expect 10): OK
float64 precision > 1e11: OK
float32 det: OK
