--TEST--
float128 arithmetic: result dtype, precision, and toArray returns strings
--FILE--
<?php
$b = NumPower::array([['1.01234567890321654987', 2],[3, 3]], 'float128');
$c = NumPower::array([['0.123456789987654321', 2],[3, 3]], 'float128');

// Addition
$x = $b + $c;
$arr = $x->toArray();
echo $arr[0][0] . "\n";
echo $arr[0][1] . "\n";
echo $arr[1][0] . "\n";
echo $arr[1][1] . "\n";

// toArray returns strings for float128 (PHP float cannot represent full precision)
var_dump(gettype($arr[0][0]) === 'string');

// Subtraction: 2 - 2 = 0
$s = $b - $c;
echo $s->toArray()[0][1] . "\n";

// Multiplication: 2 * 2 = 4
$m = $b * $c;
echo $m->toArray()[0][1] . "\n";

// Division: 2 / 2 = 1
$d = $b / $c;
echo $d->toArray()[0][1] . "\n";
?>
--EXPECT--
1.13580246889087087
4
6
6
bool(true)
0
4
1
