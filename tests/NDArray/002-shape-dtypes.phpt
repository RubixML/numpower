--TEST--
NDArray::shape() returns correct shape for all supported data types
--FILE--
<?php
$dtypes = ['float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

foreach ($dtypes as $dtype) {
    $a = NumPower::array([[1, 2, 3], [4, 5, 6]], $dtype);
    echo $dtype . ':';
    print_r($a->shape());
}
?>
--EXPECT--
float32:Array
(
    [0] => 2
    [1] => 3
)
float64:Array
(
    [0] => 2
    [1] => 3
)
int8:Array
(
    [0] => 2
    [1] => 3
)
uint8:Array
(
    [0] => 2
    [1] => 3
)
int16:Array
(
    [0] => 2
    [1] => 3
)
uint16:Array
(
    [0] => 2
    [1] => 3
)
int32:Array
(
    [0] => 2
    [1] => 3
)
uint32:Array
(
    [0] => 2
    [1] => 3
)
int64:Array
(
    [0] => 2
    [1] => 3
)
uint64:Array
(
    [0] => 2
    [1] => 3
)
