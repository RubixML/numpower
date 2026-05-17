--TEST--
$a[i] = $scalar on a 2-D NDArray broadcasts the scalar across the row, dtype-correct
--FILE--
<?php
/* For a 2-D target, $a[i] is a 1-D slice; assigning a scalar fills the
   whole row with that scalar in the target's dtype. */

$cases = [
    /* dtype     init                                   row, val,                    expected post-row */
    ['int32',    [[1,2,3],[4,5,6]],                     1,    99,                     [99,99,99]],
    ['int64',    [[1,2,3],[4,5,6]],                     1,    PHP_INT_MAX,            [PHP_INT_MAX, PHP_INT_MAX, PHP_INT_MAX]],
    ['float32',  [[1.0,2.0,3.0],[4.0,5.0,6.0]],         0,    1.5,                    [1.5,1.5,1.5]],
    ['float64',  [[1.0,2.0,3.0],[4.0,5.0,6.0]],         0,    1.5,                    [1.5,1.5,1.5]],
];
foreach ($cases as [$t, $d, $row, $val, $exp_row]) {
    $a = new NDArray($d, $t);
    $a[$row] = $val;
    $got = $a[$row]->toArray();
    echo "$t row[$row]=$val: ", ($got === $exp_row ? "OK" : "BAD got=" . var_export($got, true)), "\n";
}

/* String-based scalar broadcast for string-IO dtypes */
$a = new NDArray([['1','2'],['3','4']], 'float128');
$a[1] = '3.14159265358979324';
$got = $a[1]->toArray();
echo "float128 row[1]='3.14159265358979324': ", ($got === ['3.14159265358979324','3.14159265358979324'] ? "OK" : "BAD got=" . var_export($got, true)), "\n";

$a = new NDArray([['1','2'],['3','4']], 'uint64');
$a[1] = '18446744073709551615';
$got = $a[1]->toArray();
echo "uint64 row[1]='18446744073709551615': ", ($got === ['18446744073709551615','18446744073709551615'] ? "OK" : "BAD got=" . var_export($got, true)), "\n";

/* NDArray-source assignment with matching dtype */
$a = new NDArray([[1,2,3],[4,5,6]], 'int64');
$row = new NDArray([100,200,300], 'int64');
$a[0] = $row;
$got = $a->toArray();
echo "int64 row[0]=NDArray: ", ($got === [[100,200,300],[4,5,6]] ? "OK" : "BAD got=" . var_export($got, true)), "\n";
?>
--EXPECT--
int32 row[1]=99: OK
int64 row[1]=9223372036854775807: OK
float32 row[0]=1.5: OK
float64 row[0]=1.5: OK
float128 row[1]='3.14159265358979324': OK
uint64 row[1]='18446744073709551615': OK
int64 row[0]=NDArray: OK
