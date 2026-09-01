--TEST--
NumPower::matmul float64 (double-precision matrix multiply)
--FILE--
<?php
use NumPower as nd;

/* float64 matmul used to route through a float32 BLAS call that treated the
   8-byte double buffers as 4-byte float buffers, producing subnormal or
   wildly out-of-range products. The result dtype was also declared float32,
   so float64 users got float32 values with float64 shapes.
   Float64 matmul must go through double-precision BLAS (cblas_dgemm /
   cublasDgemm) and the output must remain float64. */

$A = nd::array([[1.0, 2.0], [3.0, 4.0]], 'float64');
$B = nd::array([[5.0, 6.0], [7.0, 8.0]], 'float64');

$C = nd::matmul($A, $B);
$got = $C->toArray();

/* Expected: [[19, 22], [43, 50]]  — exact in float64 */
$ok = (
    abs($got[0][0] - 19) < 1e-15 &&
    abs($got[0][1] - 22) < 1e-15 &&
    abs($got[1][0] - 43) < 1e-15 &&
    abs($got[1][1] - 50) < 1e-15
);
echo "values: ", $ok ? "OK" : "BAD " . json_encode($got), "\n";

$ser = $C->__serialize();
echo "dtype stays float64: ", ($ser['dtype'] === 'float64') ? "OK" : "BAD got=" . $ser['dtype'], "\n";

/* A non-trivial case where float32 would visibly diverge from float64. */
$A2 = nd::array([[1234.5678, 2345.6789], [3456.7890, 4567.8901]], 'float64');
$B2 = nd::array([[11.111111, 22.222222], [33.333333, 44.444444]], 'float64');
$C2 = nd::matmul($A2, $B2);
$got2 = $C2->toArray();
/* Compute in double: each entry is 2-term inner product of exact double inputs */
$exp2 = [
    [1234.5678*11.111111 + 2345.6789*33.333333,
     1234.5678*22.222222 + 2345.6789*44.444444],
    [3456.7890*11.111111 + 4567.8901*33.333333,
     3456.7890*22.222222 + 4567.8901*44.444444],
];
$ok2 = true;
for ($i = 0; $i < 2; $i++) {
    for ($j = 0; $j < 2; $j++) {
        if (abs($got2[$i][$j] - $exp2[$i][$j]) > 1e-12) $ok2 = false;
    }
}
echo "float64 precision: ", $ok2 ? "OK" : "BAD \n got: " . json_encode($got2), "\n";
?>
--EXPECT--
values: OK
dtype stays float64: OK
float64 precision: OK
