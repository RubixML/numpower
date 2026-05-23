--TEST--
NumPower::array accepts sparse / non-zero keys at every nesting level (2-D, 3-D, CPU)
--FILE--
<?php
/* The intake walks first-present-element at every depth (not index 0).
 * Verify that sparse keys at outer / inner / both levels all yield the
 * correct shape and values. */

/* 2-D — outer sparse, inner packed */
$m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
unset($m1[0]);  /* outer keys: 1, 2 */
$a1 = NumPower::array($m1, 'float32');
echo "outer_sparse shape: ", json_encode($a1->shape()), "\n";
echo "outer_sparse value:\n", $a1, "\n";

/* 2-D — outer packed, all inner rows sparse but consistent */
$m2 = [[9, 1, 2], [9, 3, 4]];
unset($m2[0][0]);
unset($m2[1][0]);
$a2 = NumPower::array($m2, 'float64');
echo "inner_sparse shape: ", json_encode($a2->shape()), "\n";
echo "inner_sparse value:\n", $a2, "\n";

/* 2-D — sparse at both levels */
$m3 = [[9, 1, 2], [9, 3, 4], [9, 5, 6]];
unset($m3[0]);
foreach ($m3 as $k => $v) { unset($m3[$k][0]); }
$a3 = NumPower::array($m3, 'int32');
echo "both_sparse shape: ", json_encode($a3->shape()), "\n";
echo "both_sparse value:\n", $a3, "\n";

/* 3-D — outer sparse, others packed */
$t = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]];
unset($t[0]);
$a4 = NumPower::array($t, 'int16');
echo "3d_outer_sparse shape: ", json_encode($a4->shape()), "\n";
echo "3d_outer_sparse value:\n", $a4, "\n";

/* 2-D — string keys on the inner rows */
$m4 = [
    ['x' => 1, 'y' => 2, 'z' => 3],
    ['x' => 4, 'y' => 5, 'z' => 6],
];
$a5 = NumPower::array($m4, 'float32');
echo "inner_string_keys shape: ", json_encode($a5->shape()), "\n";
echo "inner_string_keys value:\n", $a5, "\n";

/* 3-D — sparse outer with non-uniform shape would be ragged → must error */
try {
    $bad = [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]];
    NumPower::array($bad, 'float32');
    echo "ragged_3d: NO ERROR\n";
} catch (\Throwable $e) {
    echo "ragged_3d: caught\n";
}
?>
--EXPECT--
outer_sparse shape: [2,3]
outer_sparse value:
[[4, 5, 6]
 [7, 8, 9]]
inner_sparse shape: [2,2]
inner_sparse value:
[[1, 2]
 [3, 4]]
both_sparse shape: [2,2]
both_sparse value:
[[3, 4]
 [5, 6]]
3d_outer_sparse shape: [2,2,2]
3d_outer_sparse value:
[[[5, 6]
  [7, 8]]
[[9, 10]
  [11, 12]]]
inner_string_keys shape: [2,3]
inner_string_keys value:
[[1, 2, 3]
 [4, 5, 6]]
ragged_3d: caught
