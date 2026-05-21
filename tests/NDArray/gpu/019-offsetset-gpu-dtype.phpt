--TEST--
$a[i] = $scalar on a GPU NDArray preserves precision and stays on GPU
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Scalar offsetSet on a GPU array writes the value through a host-side
   typed buffer and one cudaMemcpy, preserving dtype precision and leaving
   the array on the GPU. */

$cases = [
    ['int8',     [1,2,3,4], 1, 127,            127],
    ['uint8',    [1,2,3,4], 2, 255,            255],
    ['int32',    [1,2,3,4], 1, 2147483647,     2147483647],
    ['int64',    [1,2,3,4], 1, PHP_INT_MAX,    PHP_INT_MAX],
    ['float32',  [1.0,2.0,3.0], 1, 1.5,        1.5],
    ['float64',  [1.0,2.0,3.0], 1, 1.5,        1.5],
];
foreach ($cases as [$t, $init, $idx, $val, $exp]) {
    $a = (new NDArray($init, $t))->gpu();
    $a[$idx] = $val;
    /* Read back from GPU */
    $got = $a[$idx];
    $on_gpu = $a->isGPU();
    echo "$t [$idx]=$val: ", ($got === $exp ? "OK" : "BAD got=" . var_export($got, true)),
         " gpu=", ($on_gpu ? "yes" : "no"), "\n";
}

/* String values for float128 / uint64. fp128 on GPU is stored as
   double-double, so use a value exactly representable in fp64 for round-
   trip parity. */
$a = (new NDArray(['1','2','3'], 'float128'))->gpu();
$a[1] = '1.25';
echo "float128 GPU [1]='1.25': ",
     ($a[1] === '1.25' ? "OK" : "BAD got=" . var_export($a[1], true)), "\n";

$a = (new NDArray(['1','2','3'], 'uint64'))->gpu();
$a[1] = '18446744073709551615';
echo "uint64 GPU [1]='18446744073709551615': ",
     ($a[1] === '18446744073709551615' ? "OK" : "BAD got=" . var_export($a[1], true)), "\n";
?>
--EXPECT--
int8 [1]=127: OK gpu=yes
uint8 [2]=255: OK gpu=yes
int32 [1]=2147483647: OK gpu=yes
int64 [1]=9223372036854775807: OK gpu=yes
float32 [1]=1.5: OK gpu=yes
float64 [1]=1.5: OK gpu=yes
float128 GPU [1]='1.25': OK
uint64 GPU [1]='18446744073709551615': OK
