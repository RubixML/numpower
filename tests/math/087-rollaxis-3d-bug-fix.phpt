--TEST--
NDArray::transpose on 3-D arrays with axis permutation (Rollaxis bug fix)
--FILE--
<?php
/* The `NDArray_Rollaxis` axis-shift loop wrote past the allocated buffer
   and shifted the wrong range, producing a permutation vector with a
   repeated axis (e.g. `rollaxis(a, 1, 0)` on a 3-D array generated
   `[1, 0, 1]` instead of `[1, 0, 2]`). `NDArray_Transpose` then threw
   "repeated axis in transpose".
   The fix shifts only `[start..axis-1]` right by one (or the matching
   left shift for `start > axis`) and uses an inclusive `start > n`
   bound check instead of the broken `!(0 <= start < n+1)` chained
   comparison. */

/* Build a 3-D NDArray and exercise rollaxis through transpose. */
$a = NumPower::array([[[0,1,2,3], [4,5,6,7], [8,9,10,11]],
                      [[12,13,14,15], [16,17,18,19], [20,21,22,23]]],
                     'int32');
/* shape (2, 3, 4) */

/* Transpose to [1, 0, 2] — moves axis 1 to position 0 — matches
   numpy rollaxis(a, 1, 0). */
$t = NumPower::transpose($a, [1, 0, 2]);
echo "shape: ", json_encode($t->shape()), "\n";  // expect [3, 2, 4]
echo "[0]: ",  json_encode($t[0]->toArray()),  "\n";
echo "[1]: ",  json_encode($t[1]->toArray()),  "\n";

/* 3-D axis-1 sum exercises the same code path internally. */
$s = NumPower::sum($a, 1);
echo "sum axis=1 shape: ", json_encode($s->shape()), "\n";  // expect [2, 4]
echo "sum axis=1 vals: ", json_encode($s->toArray()), "\n";
/* Expected: per-axis-1 sums:
    a[0,*,j] sum = (0+4+8, 1+5+9, 2+6+10, 3+7+11) = (12, 15, 18, 21)
    a[1,*,j] sum = (12+16+20, 13+17+21, 14+18+22, 15+19+23) = (48, 51, 54, 57) */

/* 4-D rollaxis through 3-axis permutation. */
$b = NumPower::array(
    [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]],
    'float32');
/* shape (2, 2, 2, 2) */
/* Sum on axis=2 — exercises rollaxis(2, 0). */
$s2 = NumPower::sum($b, 2);
echo "4D sum axis=2 shape: ", json_encode($s2->shape()), "\n";  // expect [2,2,2]

echo "ok\n";
?>
--EXPECT--
shape: [3,2,4]
[0]: [[0,1,2,3],[12,13,14,15]]
[1]: [[4,5,6,7],[16,17,18,19]]
sum axis=1 shape: [2,4]
sum axis=1 vals: [[12,15,18,21],[48,51,54,57]]
4D sum axis=2 shape: [2,2,2]
ok
