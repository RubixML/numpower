--TEST--
NumPower::solve float64 (double-precision solver)
--FILE--
<?php
use NumPower as nd;

/* float64 solve passed ldb=n (rows of b) to LAPACKE dgesv under the
   row-major convention. Correct ldb is nrhs (columns of row-major b).
   Wrong ldb produced the silently-wrong first element:
     diag(2,3) b=[5,7]  ->  [2.5, 7] (expect [2.5, 7/3])
     [[2,3],[4,5]] b=[1,2] -> [4.97e19, 2] (expect [0.5, 0]) */

/* diag(2,3) b=[5,7]: x = [2.5, 7/3] */
$a = nd::array([[2.0, 0.0], [0.0, 3.0]], "float64");
$b = nd::array([[5.0], [7.0]], "float64");
$r = nd::solve($a, $b);
$v = $r->toArray();
$ok1 = abs($v[0][0] - 2.5) < 1e-15 && abs($v[1][0] - 7.0/3.0) < 1e-15;
echo "diag(2,3) b=[5,7]: ", $ok1 ? "OK" : "BAD " . json_encode($v), "\n";

/* [[2,3],[4,5]] det=-2, inv = [[-2.5, 1.5],[2, -1]], b=[1,2] -> [0.5, 0] */
$a2 = nd::array([[2.0, 3.0], [4.0, 5.0]], "float64");
$b2 = nd::array([[1.0], [2.0]], "float64");
$r2 = nd::solve($a2, $b2);
$v2 = $r2->toArray();
$ok2 = abs($v2[0][0] - 0.5) < 1e-15 && abs($v2[1][0] - 0.0) < 1e-15;
echo "[[2,3],[4,5]] b=[1,2]: ", $ok2 ? "OK" : "BAD " . json_encode($v2), "\n";

$d = $r->__serialize()['dtype'];
echo "dtype stays float64: ", ($d === 'float64') ? "OK" : "BAD $d", "\n";
?>
--EXPECT--
diag(2,3) b=[5,7]: OK
[[2,3],[4,5]] b=[1,2]: OK
dtype stays float64: OK
