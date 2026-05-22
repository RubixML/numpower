--TEST--
NDArray::count() returns the same axis-0 length for every supported dtype
--FILE--
<?php
/* count() reads metadata only; it must be independent of the storage dtype. */
$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];

foreach ($dtypes as $dtype) {
    /* 2x3 matrix -> count == 2 for every dtype. */
    $a = NumPower::array([[1, 2, 3], [4, 5, 6]], $dtype);
    echo $dtype . ':' . $a->count() . PHP_EOL;
}

foreach ($dtypes as $dtype) {
    /* 1-D length 7. */
    $a = NumPower::array([1, 2, 3, 4, 5, 6, 7], $dtype);
    echo $dtype . '-1d:' . $a->count() . PHP_EOL;
}

foreach ($dtypes as $dtype) {
    /* 0-D scalar -> 0 (no axis 0). */
    $a = new NDArray(1, $dtype);
    echo $dtype . '-0d:' . $a->count() . PHP_EOL;
}
?>
--EXPECT--
float4:2
float8:2
float16:2
float32:2
float64:2
float128:2
int8:2
uint8:2
int16:2
uint16:2
int32:2
uint32:2
int64:2
uint64:2
float4-1d:7
float8-1d:7
float16-1d:7
float32-1d:7
float64-1d:7
float128-1d:7
int8-1d:7
uint8-1d:7
int16-1d:7
uint16-1d:7
int32-1d:7
uint32-1d:7
int64-1d:7
uint64-1d:7
float4-0d:0
float8-0d:0
float16-0d:0
float32-0d:0
float64-0d:0
float128-0d:0
int8-0d:0
uint8-0d:0
int16-0d:0
uint16-0d:0
int32-0d:0
uint32-0d:0
int64-0d:0
uint64-0d:0
