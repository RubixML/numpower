--TEST--
NumPower::inv float64 (double-precision inverse)
--FILE--
<?php
use NumPower as nd;

/* float64 inverse must be computed in double precision. Before the fix the
   float32 path (matrixFloatInverse / sgetrf) was used unconditionally, which
   re-cast 8-byte doubles as 4-byte floats and produced ~1e-8 magnitude error
   against the true double inverse below. The tolerance of at-most-1e-8 lets a
   float64 result pass while a float32-computed result (error ~3.4e-8 here)
   fails. */
$A = nd::array([
    [2.5, 3.1, 1.7],
    [0.9, 4.2, 5.6],
    [6.3, 2.1, 0.8],
], 'float64');

$Ainv = nd::inv($A);

/* Result must stay float64 (dtype is a first-class property of the result) */
$ser = $Ainv->__serialize();
echo "output dtype: ", $ser['dtype'], "\n";
echo "dtype stays float64: ", ($ser['dtype'] === 'float64') ? "OK" : "BAD", "\n";

$got = $Ainv->toArray();
$expected = [
    [-0.18932990736358096, 0.024567809407893233, 0.2303513872923569],
    [0.778957333153019, -0.19631708251628455, -0.2810647553361732],
    [-0.5537899790384745, 0.3218608425180877, 0.17377780783014402],
];

$max_err = 0.0;
for ($i = 0; $i < 3; $i++) {
    for ($j = 0; $j < 3; $j++) {
        $e = abs($got[$i][$j] - $expected[$i][$j]);
        if ($e > $max_err) {
            $max_err = $e;
        }
    }
}
echo "float64 precision (<1e-8): ", ($max_err < 1e-8) ? "OK" : "BAD (max err = " . $max_err . ")", "\n";
?>
--EXPECT--
output dtype: float64
dtype stays float64: OK
float64 precision (<1e-8): OK
