--TEST--
NDArray::size() returns correct element count for all supported data types
--FILE--
<?php
$dtypes = ['float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($dtypes as $dtype) {
    $a = NumPower::array([[1, 2, 3], [4, 5, 6]], $dtype);
    echo $dtype . ':' . $a->size() . PHP_EOL;
}
?>
--EXPECT--
float32:6
float64:6
int8:6
uint8:6
int16:6
uint16:6
int32:6
uint32:6
int64:6
uint64:6
