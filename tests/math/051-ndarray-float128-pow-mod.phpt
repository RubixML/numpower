--TEST--
float128 pow and mod: use quadmath (powq/fmodq) when available, else long double fallback
--FILE--
<?php
$two   = NumPower::array(['2'], 'float128');
$three = NumPower::array(['3'], 'float128');
$seven = NumPower::array(['7'], 'float128');

// 2 ** 3 = 8
$p = NumPower::pow($two, $three);
echo $p->toArray()[0] . "\n";

// 7 % 3 = 1
$m = NumPower::mod($seven, $three);
echo $m->toArray()[0] . "\n";

// Result type must be float128 (toArray returns strings)
var_dump(gettype($p->toArray()[0]) === 'string');
var_dump(gettype($m->toArray()[0]) === 'string');
?>
--EXPECT--
8
1
bool(true)
bool(true)
