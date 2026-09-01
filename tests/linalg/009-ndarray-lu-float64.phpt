--TEST--
NumPower::lu float64 (double-precision LU decomposition)
--FILE--
<?php
use NumPower as nd;

/* float64 LU previously fell through to the float32 path (treated double
   arrays as float32) which destroyed the values. It must use the double
   partial-pivoting LU (matrixDoubleLU / d*getrf) and keep the result in
   float64. Verify P, L, U are consistent: P·A == L·U within 1e-12, det
   consistent, and U is upper-triangular. */

$A = [[4.0, 3.0], [6.0, 3.0]];
$lu = nd::lu(nd::array($A, "float64"));
$P = $lu[0]->toArray();
$L = $lu[1]->toArray();
$U = $lu[2]->toArray();

/* P must be a permutation matrix: one 1 per row and column */
$perm_ok = ($P[0][0] + $P[0][1] === 1.0) && ($P[1][0] + $P[1][1] === 1.0)
        && ($P[0][0] + $P[1][0] === 1.0) && ($P[0][1] + $P[1][1] === 1.0);
echo "P is permutation: ", $perm_ok ? "OK" : "BAD " . json_encode($P), "\n";

/* U must be upper-triangular (within 1e-12 of 0) */
$u_ok = abs($U[1][0]) < 1e-12;
echo "U upper-triangular: ", $u_ok ? "OK" : "BAD " . json_encode($U), "\n";

/* L must be lower-triangular with unit diagonal */
$l_ok = (abs($L[0][1]) < 1e-12 && abs($L[0][0] - 1.0) < 1e-15
      && abs($L[1][1] - 1.0) < 1e-15);
echo "L lower-tri diag-1: ", $l_ok ? "OK" : "BAD " . json_encode($L), "\n";

/* P · A == L · U */
$PA = [];
for ($i = 0; $i < 2; $i++) for ($j = 0; $j < 2; $j++) {
    $PA[$i][$j] = $P[$i][0] * $A[0][$j] + $P[$i][1] * $A[1][$j];
}
$LU = [];
for ($i = 0; $i < 2; $i++) for ($j = 0; $j < 2; $j++) {
    $s = 0.0;
    for ($k = 0; $k < 2; $k++) $s += $L[$i][$k] * $U[$k][$j];
    $LU[$i][$j] = $s;
}
$ok = true;
for ($i = 0; $i < 2; $i++) for ($j = 0; $j < 2; $j++) {
    if (abs($PA[$i][$j] - $LU[$i][$j]) > 1e-12) $ok = false;
}
echo "P·A == L·U: ", $ok ? "OK" : "BAD PA=" . json_encode($PA) . " LU=" . json_encode($LU), "\n";

/* determinant: det(A) = det(P) * det(L) * det(U).  det(L)=1, det(P)=±1, det(U)=prod(U[diag]). */
$detP = $P[0][0] * $P[1][1] - $P[0][1] * $P[1][0];
$detU = $U[0][0] * $U[1][1];
$calc = $detP * $detU;
$expect = 4.0*3.0 - 6.0*3.0;   // -6
$det_ok = abs($calc - $expect) < 1e-12;
echo "det(L)=1, det(P)*det(U) matches det(A) ($expect): ", $det_ok ? "OK" : "BAD calc=$calc", "\n";

$dtypes = array_map(fn($x) => $x->__serialize()['dtype'], $lu);
echo "dtypes all float64: ", (count(array_unique($dtypes)) === 1 && $dtypes[0] === 'float64') ? "OK" : "BAD " . json_encode($dtypes), "\n";
?>
--EXPECT--
P is permutation: OK
U upper-triangular: OK
L lower-tri diag-1: OK
P·A == L·U: OK
det(L)=1, det(P)*det(U) matches det(A) (-6): OK
dtypes all float64: OK
