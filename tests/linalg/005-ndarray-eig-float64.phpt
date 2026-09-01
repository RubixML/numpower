--TEST--
NumPower::eig float64 (double-precision eigenvalues/eigenvectors)
--FILE--
<?php
use NumPower as nd;

/* float64 eig used to fall through to the float32 path; it must use dgeev
   and keep the result in float64. rtn[0] = eigenvalues, rtn[1] = eigenvectors. */

$rtn = nd::eig(nd::array([[2.0, 0.0], [0.0, 3.0]], "float64"));
$eigenvalues  = $rtn[0]->toArray();
$eigenvectors = $rtn[1]->toArray();

/* eigenvalues of diag(2,3) are [2, 3] (order can vary by LAPACK). */
$vals = array_values((function($a) {
    $out = [];
    array_walk_recursive($a, function($v) use (&$out) { $out[] = (float)$v; });
    return $out;
})($eigenvalues));
sort($vals);
$ok1 = (abs($vals[0] - 2.0) < 1e-15 && abs($vals[1] - 3.0) < 1e-15);
echo "eigenvalues diag(2,3): ", $ok1 ? "OK" : "BAD " . json_encode($vals), "\n";

/* eigenvalues of [[4,1],[2,3]] are 2 and 5 (char poly lambda^2 - 7 lambda + 10 = 0). */
$rtn2 = nd::eig(nd::array([[4.0, 1.0], [2.0, 3.0]], "float64"));
$vals2 = array_values((function($a) {
    $out = [];
    array_walk_recursive($a, function($v) use (&$out) { $out[] = (float)$v; });
    return $out;
})($rtn2[0]->toArray()));
sort($vals2);
$ok2 = (abs($vals2[0] - 2.0) < 1e-14 && abs($vals2[1] - 5.0) < 1e-14);
echo "eigenvalues [[4,1],[2,3]]: ", $ok2 ? "OK" : "BAD " . json_encode($vals2), "\n";

/* dtype stays float64 in both outputs (rtn[0] eigenvalues, rtn[1] eigenvectors) */
$d0 = $rtn[0]->__serialize()['dtype'];
$d1 = $rtn[1]->__serialize()['dtype'];
echo "dtypes float64: ", ($d0 === 'float64' && $d1 === 'float64') ? "OK" : "BAD $d0/$d1", "\n";
?>
--EXPECT--
eigenvalues diag(2,3): OK
eigenvalues [[4,1],[2,3]]: OK
dtypes float64: OK
