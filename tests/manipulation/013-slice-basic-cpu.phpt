--TEST--
NumPower::slice() basic semantics on CPU: dim reduction, range, step, negative indices, defaults
--FILE--
<?php
/* Verifies the four slice-spec forms (single int, [], [start,stop], [start,stop,step]),
   negative indices for both single ints and ranges, dim reduction on single int,
   leading-axis vs trailing-axis slicing, and partial axis spec (fewer than NDIM).
   Uses the non-mutating static form so the same source can be reused for several
   sub-tests. The mutating instance method is covered separately in 022. */

$m = NumPower::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

/* docstring examples */
echo "row0:     ", NumPower::slice($m, 0),         "\n";   /* [1, 2, 3] */
echo "lastCol:  ", NumPower::slice($m, [], -1),    "\n";   /* [3, 6]    */
echo "sub:      ", NumPower::slice($m, 0, [0, 2]), "\n";   /* [1, 2]    */

/* slice on 1D returns a 0-D scalar (float) */
$v = NumPower::array([10.0, 20.0, 30.0, 40.0]);
$s = NumPower::slice($v, 0);
echo "1D slice(0) type=", gettype($s), " val=", $s, "\n";
echo "1D slice(-1) val=", NumPower::slice($v, -1), "\n";

/* range with [start, stop] only — step defaults to 1 */
$r = NumPower::arange(10.0);
echo "range[2,6]:  ", NumPower::slice($r, [2, 6]), "\n";   /* [2, 3, 4, 5] */
echo "range[,5]:   ", NumPower::slice($r, [0, 5]), "\n";   /* [0, 1, 2, 3, 4] */
echo "range[5,]:   ", NumPower::slice($r, [5, 10]), "\n";  /* [5, 6, 7, 8, 9] */

/* step != 1 */
echo "step[1,9,2]: ", NumPower::slice($r, [1, 9, 2]),  "\n";
echo "step[0,10,3]:", NumPower::slice($r, [0, 10, 3]), "\n";
echo "step[9,0,-2]:", NumPower::slice($r, [9, 0, -2]), "\n";
echo "step[-1,-6,-1]:", NumPower::slice($r, [-1, -6, -1]), "\n";

/* shapes */
print_r(NumPower::slice($m, 0)->shape());
print_r(NumPower::slice($m, [], -1)->shape());
print_r(NumPower::slice($m, 0, [0, 2])->shape());

/* 3D — partial axis spec, dim reduction, retention of trailing axes */
$t3 = NumPower::array([
  [[ 0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9,10,11]],
  [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
]);
print_r(NumPower::slice($t3, 0)->shape());
print_r(NumPower::slice($t3, [], 0)->shape());
print_r(NumPower::slice($t3, [], [], 0)->shape());
print_r(NumPower::slice($t3, [0, 2], [1, 3])->shape());
echo "slice(1)[0][3] = ", NumPower::slice($t3, 1)[0][3], "\n";
echo "slice(0,1,2) = ", NumPower::slice($t3, 0, 1, 2), "\n";
?>
--EXPECT--
row0:     [1, 2, 3]
lastCol:  [3, 6]
sub:      [1, 2]
1D slice(0) type=double val=10
1D slice(-1) val=40
range[2,6]:  [2, 3, 4, 5]
range[,5]:   [0, 1, 2, 3, 4]
range[5,]:   [5, 6, 7, 8, 9]
step[1,9,2]: [1, 3, 5, 7]
step[0,10,3]:[0, 3, 6, 9]
step[9,0,-2]:[9, 7, 5, 3, 1]
step[-1,-6,-1]:[9, 8, 7, 6, 5]
Array
(
    [0] => 3
)
Array
(
    [0] => 2
)
Array
(
    [0] => 2
)
Array
(
    [0] => 3
    [1] => 4
)
Array
(
    [0] => 2
    [1] => 4
)
Array
(
    [0] => 2
    [1] => 3
)
Array
(
    [0] => 2
    [1] => 2
    [2] => 4
)
slice(1)[0][3] = 15
slice(0,1,2) = 6
