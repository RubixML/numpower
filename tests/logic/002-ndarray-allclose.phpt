--TEST--
NDArray::allClose
--FILE--
<?php
use \NDArray as nd;

$a = nd::array([[1, 2], [3, 4]]);
$b = nd::array([[5, 6], [7, 8]]);
$c = nd::array([9, 10]);
var_dump(nd::allClose($a, $b));
var_dump(nd::allClose($a, $a));
var_dump(nd::allClose($c, $c));
?>
--EXPECT--
bool(false)
bool(true)
bool(true)