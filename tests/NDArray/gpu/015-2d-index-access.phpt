--TEST--
2-D index access on GPU: $a[i][j] returns the typed scalar matching CPU
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Two-level index access on a GPU NDArray exercises the path:
   $a[i] -> 1-D slice with base=parent (GPU data),
   $a[i][j] -> 0-D slice -> NDArray_ScalarToZval -> cudaMemcpy of one element.
   The result must equal the CPU read of the same element. */

$cases = [
    'float32'  => [[0.0, 1.5, -1.5], [1e3, 1e-3, -1e3]],
    'float64'  => [[0.0, 1.5, -1.5], [1e3, 1e-3, -1e3]],
    'float128' => [['0', '1.5', '-1.5'], ['1e3', '1e-9', '-1e3']],
    'int8'     => [[-128, -1, 0], [1, 64, 127]],
    'int32'    => [[-2147483648, 0, 1], [2147483647, -1, 7]],
    'int64'    => [[PHP_INT_MIN, 0, 1], [PHP_INT_MAX, -1, 7]],
    'uint64'   => [['0', '1', '2'], ['18446744073709551613', '18446744073709551614', '18446744073709551615']],
];

foreach ($cases as $dtype => $data) {
    $cpu = new NDArray($data, $dtype);
    $gpu = $cpu->gpu();
    $rows = count($data);
    $cols = count($data[0]);
    $ok = true;
    for ($i = 0; $i < $rows; $i++) {
        for ($j = 0; $j < $cols; $j++) {
            $cv = $cpu[$i][$j];
            $gv = $gpu[$i][$j];
            if ($cv !== $gv) { $ok = false; break 2; }
        }
    }
    echo "$dtype: ", ($ok ? "OK" : "BAD"), "\n";
}
?>
--EXPECT--
float32: OK
float64: OK
float128: OK
int8: OK
int32: OK
int64: OK
uint64: OK
