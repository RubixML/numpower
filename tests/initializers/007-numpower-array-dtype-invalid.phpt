--TEST--
NumPower::array with invalid dtype throws an error
--FILE--
<?php
try {
    $a = NumPower::array([1, 2, 3], 'badtype');
} catch (Error $e) {
    echo $e->getMessage() . "\n";
}
?>
--EXPECT--
Invalid data type 'badtype'. Supported: float4, float8, float16, float32, float64, float128, int8, uint8, int16, uint16, int32, uint32, int64, uint64
