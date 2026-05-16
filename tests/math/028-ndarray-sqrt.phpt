--TEST--
NumPower::sqrt
--FILE--
<?php
$a = NumPower::array([[-156, 150], [19, -39]]);
$r = NumPower::sqrt($a)->toArray();
var_dump(is_nan($r[0][0]));
echo $r[0][1] . "\n";
echo $r[1][0] . "\n";
var_dump(is_nan($r[1][1]));

$r2 = NumPower::sqrt($a[0])->toArray();
var_dump(is_nan($r2[0]));
echo $r2[1] . "\n";

$r3 = NumPower::sqrt([[0], [-0.5]])->toArray();
echo $r3[0][0] . "\n";
var_dump(is_nan($r3[1][0]));
?>
--EXPECTF--
bool(true)
%f
%f
bool(true)
bool(true)
%f
0
bool(true)
