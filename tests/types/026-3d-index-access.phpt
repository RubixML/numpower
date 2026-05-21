--TEST--
3-D index access $a[i][j][k] returns the correct typed scalar across dtypes
--FILE--
<?php
/* Three-level slice + read exercises the iterator chain twice before the
   final scalar conversion. Any drift in dtype propagation would manifest
   here as wrong PHP types or wrong values. */

$cases = [
    'float64'  => [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    'float128' => [[['1','2'],['3','4']],[['5','6'],['7','8']]],
    'int8'     => [[[-128, -1], [0, 1]], [[64, 127], [-64, 63]]],
    'int64'    => [[[PHP_INT_MIN, 0], [1, 2]], [[3, 4], [5, PHP_INT_MAX]]],
    'uint64'   => [[['0','1'],['2','3']],[['4','5'],['6','18446744073709551615']]],
];

foreach ($cases as $t => $d) {
    $a = new NDArray($d, $t);
    $arr = $a->toArray();
    $ok = true;
    /* Compare every element via index access against the toArray representation. */
    for ($i = 0; $i < 2; $i++) {
        for ($j = 0; $j < 2; $j++) {
            for ($k = 0; $k < 2; $k++) {
                if ($a[$i][$j][$k] !== $arr[$i][$j][$k]) { $ok = false; break 3; }
            }
        }
    }
    echo "$t: ", ($ok ? "OK" : "BAD"), "\n";
}
?>
--EXPECT--
float64: OK
float128: OK
int8: OK
int64: OK
uint64: OK
