--TEST--
NumPower::svd float64 (double-precision SVD)
--FILE--
<?php
use NumPower as nd;

/* float64 SVD previously read uninitialised double* pointers (a crash/UB
   bug). It must go through LAPACKE_dgesdd and produce finite results. */

$rtn = nd::svd(nd::array([[4.0, 3.0], [2.0, 5.0]], "float64"));
$U = $rtn[0]->toArray();
$S = $rtn[1]->toArray();
$V = $rtn[2]->toArray();

$finite = true;
array_walk_recursive($U, function($v) use (&$finite) { if (is_nan($v) || is_infinite($v)) $finite = false; });
array_walk_recursive($S, function($v) use (&$finite) { if (is_nan($v) || is_infinite($v)) $finite = false; });
array_walk_recursive($V, function($v) use (&$finite) { if (is_nan($v) || is_infinite($v)) $finite = false; });
echo "U, S, V all finite: ", $finite ? "OK" : "BAD (uninitialised memory or NaN/inf)", "\n";

/* Reconstruct A = U * diag(S) * V. */
$A = [[4.0, 3.0], [2.0, 5.0]];
$rec = [];
for ($i = 0; $i < 2; $i++) {
    $rec[$i] = [];
    for ($j = 0; $j < 2; $j++) {
        $s = 0.0;
        for ($k = 0; $k < 2; $k++) {
            $s += $U[$i][$k] * $S[$k] * $V[$k][$j];
        }
        $rec[$i][$j] = $s;
    }
}
$ok = true;
for ($i = 0; $i < 2; $i++) for ($j = 0; $j < 2; $j++) {
    if (abs($rec[$i][$j] - $A[$i][$j]) > 1e-12) $ok = false;
}
echo "A = U diag(S) V: ", $ok ? "OK" : "BAD " . json_encode($rec), "\n";

$dU = $rtn[0]->__serialize()['dtype'];
$dS = $rtn[1]->__serialize()['dtype'];
$dV = $rtn[2]->__serialize()['dtype'];
echo "dtypes all float64: ", ($dU === 'float64' && $dS === 'float64' && $dV === 'float64') ? "OK" : "BAD $dU/$dS/$dV", "\n";
?>
--EXPECT--
U, S, V all finite: OK
A = U diag(S) V: OK
dtypes all float64: OK
