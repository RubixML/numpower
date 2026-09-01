--TEST--
NumPower::lstsq float64 (double-precision least-squares)
--FILE--
<?php
use NumPower as nd;

/* float64 lstsq must use double-precision LAPACKE (dgels), keep the result
   in float64, and accept a wide a (more rows than columns). */

$a = nd::array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "float64");
$b = nd::array([[5.0], [3.0], [8.0]], "float64");
$r = nd::lstsq($a, $b);
$v = $r->toArray();
$ok = abs($v[0][0] - 5.0) < 1e-12 && abs($v[1][0] - 3.0) < 1e-12;
echo "lstsq with exact solution: ", $ok ? "OK" : "BAD " . json_encode($v), "\n";

/* dtype stays float64. */
$d = $r->__serialize()['dtype'];
echo "dtype stays float64: ", ($d === 'float64') ? "OK" : "BAD $d", "\n";

/* a float32 input should also continue to work. */
$af = nd::array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "float32");
$bf = nd::array([[5.0], [3.0], [8.0]], "float32");
$rf = nd::lstsq($af, $bf);
$vf = $rf->toArray();
$ok_f = abs($vf[0][0] - 5.0) < 1e-5 && abs($vf[1][0] - 3.0) < 1e-5;
echo "float32 still works: ", $ok_f ? "OK" : "BAD " . json_encode($vf), "\n";
?>
--EXPECT--
lstsq with exact solution: OK
dtype stays float64: OK
float32 still works: OK
