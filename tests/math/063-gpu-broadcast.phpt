--TEST--
GPU non-scalar broadcast: row+matrix, col+matrix work on GPU for every dtype, stay on GPU
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU broadcast for shape-mismatched operands now uses a byte-wise gather
   kernel — works for every dtype including fp128 (dd-encoded). */

$dtypes = ['int8', 'int16', 'int32', 'int64',
           'uint8', 'uint16', 'uint32', 'uint64',
           'float16', 'float32', 'float64', 'float128'];

/* row + 2-D matrix */
echo "=== row + matrix ===\n";
foreach ($dtypes as $dt) {
    $m = (new NDArray([[1, 2, 3], [4, 5, 6]], $dt))->gpu();
    $row = (new NDArray([10, 20, 30], $dt))->gpu();
    $r = $m + $row;
    $on_gpu = $r->isGPU();
    $arr = $r->cpu()->toArray();
    $ok = (string)$arr[0][0] === '11'
       && (string)$arr[0][1] === '22'
       && (string)$arr[0][2] === '33'
       && (string)$arr[1][0] === '14'
       && (string)$arr[1][1] === '25'
       && (string)$arr[1][2] === '36';
    echo "$dt: gpu=", ($on_gpu ? 'yes' : 'no'), " vals=", ($ok ? 'OK' : 'BAD'), "\n";
}

/* col + 2-D matrix (col shape [M,1]) */
echo "\n=== col + matrix ===\n";
foreach (['int32', 'float64', 'float128'] as $dt) {
    $m = (new NDArray([[1, 2, 3], [4, 5, 6]], $dt))->gpu();
    $col = (new NDArray([[10], [20]], $dt))->gpu();
    $r = $m + $col;
    $arr = $r->cpu()->toArray();
    /* expected: [[11,12,13],[24,25,26]] */
    $ok = (string)$arr[0][0] === '11' && (string)$arr[1][2] === '26';
    echo "$dt: gpu=", ($r->isGPU() ? 'yes' : 'no'), " vals=", ($ok ? 'OK' : 'BAD'), "\n";
}

/* Shape-incompatible should throw */
echo "\n=== incompatible shapes ===\n";
try {
    $a = (new NDArray([1, 2, 3], 'int32'))->gpu();
    $b = (new NDArray([1, 2], 'int32'))->gpu();
    $r = $a + $b;
    echo "FAIL: no throw\n";
} catch (\Error $e) {
    echo "throws ok\n";
}
?>
--EXPECT--
=== row + matrix ===
int8: gpu=yes vals=OK
int16: gpu=yes vals=OK
int32: gpu=yes vals=OK
int64: gpu=yes vals=OK
uint8: gpu=yes vals=OK
uint16: gpu=yes vals=OK
uint32: gpu=yes vals=OK
uint64: gpu=yes vals=OK
float16: gpu=yes vals=OK
float32: gpu=yes vals=OK
float64: gpu=yes vals=OK
float128: gpu=yes vals=OK

=== col + matrix ===
int32: gpu=yes vals=OK
float64: gpu=yes vals=OK
float128: gpu=yes vals=OK

=== incompatible shapes ===
throws ok
