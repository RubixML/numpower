--TEST--
Arithmetic on 2-D and 3-D arrays preserves shape & dtype across all dtypes; broadcasting works (1-D + 2-D, scalar + 2-D)
--FILE--
<?php
$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

/* 2-D array + 2-D array (same shape). Use values exactly representable in
   every dtype including float4 (E2M1: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}). */
echo "=== 2-D arr+arr ===\n";
foreach ($dtypes as $dt) {
    $a = new NDArray([[1, 1], [2, 2]], $dt);
    $b = new NDArray([[1, 2], [1, 2]], $dt);
    $r = $a + $b;
    echo "$dt shape=", json_encode($r->shape()),
         " r[0][0]=", (string)$r[0][0], " r[1][1]=", (string)$r[1][1], "\n";
}

/* 2-D + scalar broadcast */
echo "\n=== 2-D arr + scalar (weak-scalar) ===\n";
foreach (['int8', 'int32', 'float32', 'float128', 'uint64'] as $dt) {
    $a = new NDArray([[1, 2], [3, 4]], $dt);
    $r = $a + 10;
    echo "$dt shape=", json_encode($r->shape()),
         " r[0][0]=", (string)$r[0][0], " r[1][1]=", (string)$r[1][1], "\n";
}

/* 3-D arithmetic */
echo "\n=== 3-D arr+arr ===\n";
$cube = [];
for ($i = 0; $i < 2; $i++) {
    for ($j = 0; $j < 2; $j++) {
        for ($k = 0; $k < 2; $k++) {
            $cube[$i][$j][$k] = $i * 4 + $j * 2 + $k + 1;
        }
    }
}
foreach (['int32', 'float64', 'float128'] as $dt) {
    $a = new NDArray($cube, $dt);
    $r = $a + $a;
    echo "$dt shape=", json_encode($r->shape()),
         " r[0][0][0]=", (string)$r[0][0][0],
         " r[1][1][1]=", (string)$r[1][1][1], "\n";
}

/* 1-D + 2-D broadcast: row broadcasts across rows */
echo "\n=== row + matrix ===\n";
$mat = new NDArray([[1, 2, 3], [4, 5, 6]], 'float32');
$row = new NDArray([10, 20, 30], 'float32');
$r = $mat + $row;
print_r($r->toArray());

/* Shape-mismatch throws */
echo "\n=== shape mismatch ===\n";
try {
    $a = new NDArray([1, 2, 3], 'float32');
    $b = new NDArray([[1, 2], [3, 4]], 'float32');
    $r = $a + $b;
    echo "FAIL: no throw\n";
} catch (\Error $e) {
    echo "throws: ", $e->getMessage(), "\n";
}
?>
--EXPECT--
=== 2-D arr+arr ===
float4 shape=[2,2] r[0][0]=2 r[1][1]=4
float8 shape=[2,2] r[0][0]=2 r[1][1]=4
float16 shape=[2,2] r[0][0]=2 r[1][1]=4
float32 shape=[2,2] r[0][0]=2 r[1][1]=4
float64 shape=[2,2] r[0][0]=2 r[1][1]=4
float128 shape=[2,2] r[0][0]=2 r[1][1]=4
int8 shape=[2,2] r[0][0]=2 r[1][1]=4
uint8 shape=[2,2] r[0][0]=2 r[1][1]=4
int16 shape=[2,2] r[0][0]=2 r[1][1]=4
uint16 shape=[2,2] r[0][0]=2 r[1][1]=4
int32 shape=[2,2] r[0][0]=2 r[1][1]=4
uint32 shape=[2,2] r[0][0]=2 r[1][1]=4
int64 shape=[2,2] r[0][0]=2 r[1][1]=4
uint64 shape=[2,2] r[0][0]=2 r[1][1]=4

=== 2-D arr + scalar (weak-scalar) ===
int8 shape=[2,2] r[0][0]=11 r[1][1]=14
int32 shape=[2,2] r[0][0]=11 r[1][1]=14
float32 shape=[2,2] r[0][0]=11 r[1][1]=14
float128 shape=[2,2] r[0][0]=11 r[1][1]=14
uint64 shape=[2,2] r[0][0]=11 r[1][1]=14

=== 3-D arr+arr ===
int32 shape=[2,2,2] r[0][0][0]=2 r[1][1][1]=16
float64 shape=[2,2,2] r[0][0][0]=2 r[1][1][1]=16
float128 shape=[2,2,2] r[0][0][0]=2 r[1][1][1]=16

=== row + matrix ===
Array
(
    [0] => Array
        (
            [0] => 11
            [1] => 22
            [2] => 33
        )

    [1] => Array
        (
            [0] => 14
            [1] => 25
            [2] => 36
        )

)

=== shape mismatch ===
throws: Can't broadcast arrays with incompatible shapes.
