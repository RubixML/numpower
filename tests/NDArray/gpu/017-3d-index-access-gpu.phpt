--TEST--
3-D index access on GPU yields the same values as CPU at boundary inputs
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* 3-D GPU iteration: $a[i] -> 2-D slice, $a[i][j] -> 1-D slice, $a[i][j][k]
   -> 0-D slice -> NDArray_ScalarToZval (cudaMemcpy of one element). At every
   step the device flag must propagate, and at the leaf the byte-correct
   scalar read must equal the CPU read. */

$cases = [
    'float64'  => [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    'float128' => [[['1.5', '2.5'], ['3.5', '4.5']], [['5.5', '6.5'], ['7.5', '8.5']]],
    'int8'     => [[[-128, -1], [0, 1]], [[64, 127], [-64, 63]]],
    'int64'    => [[[PHP_INT_MIN, 0], [1, 2]], [[3, 4], [5, PHP_INT_MAX]]],
    'uint64'   => [[['0','1'],['2','3']],[['4','5'],['6','18446744073709551615']]],
];

foreach ($cases as $t => $d) {
    $cpu = new NDArray($d, $t);
    $gpu = $cpu->gpu();
    $ok = true;
    for ($i = 0; $i < 2; $i++) {
        for ($j = 0; $j < 2; $j++) {
            for ($k = 0; $k < 2; $k++) {
                if ($cpu[$i][$j][$k] !== $gpu[$i][$j][$k]) { $ok = false; break 3; }
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
