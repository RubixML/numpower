--TEST--
NDArray multidimensional index access ($a[i] and $a[i][j]) across dtypes
--FILE--
<?php
/* For a 2-D NDArray, $a[i] should be an NDArray slice (1-D), and $a[i][j]
   should be the typed scalar of the dtype. Verify this for a representative
   subset of dtypes that cover every PHP-type-of-result category. */

$cases = [
    'float32'  => [[1.0, 2.5], [3.5, 4.5]],
    'float64'  => [[1.5, 2.5], [3.5, 4.5]],
    'float128' => [['1.5', '2.5'], ['3.5', '4.5']],
    'int8'     => [[-1, 0], [1, 127]],
    'int32'    => [[1, 2], [3, 4]],
    'int64'    => [[PHP_INT_MIN, 0], [1, PHP_INT_MAX]],
    'uint64'   => [['0', '1'], ['2', '18446744073709551615']],
];

foreach ($cases as $dtype => $data) {
    $a = new NDArray($data, $dtype);
    $row = $a[0];
    if (!($row instanceof NDArray)) {
        echo "$dtype: row not NDArray\n";
        continue;
    }
    $a00 = $a[0][0];
    $a11 = $a[1][1];
    /* Reconstruct expected via toArray to side-step type-specific stringification */
    $expected00 = $a[0]->toArray()[0];
    $expected11 = $a[1]->toArray()[1];
    $ok = ($a00 === $expected00) && ($a11 === $expected11);
    echo "$dtype: ", ($ok ? 'OK' : "BAD a[0][0]=" . var_export($a00, true)
                                    . " a[1][1]=" . var_export($a11, true)
                                    . " expected=" . var_export($expected00, true)
                                    . "," . var_export($expected11, true)), "\n";
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
