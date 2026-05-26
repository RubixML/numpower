--TEST--
NumPower::sum
--FILE--
<?php
/* NumPower::array() defaults to float32. The sum's result dtype follows the
   input — float32 — so the returned PHP float carries float32 precision
   (~7 decimal digits, then promoted to double for display).
   Prior to dtype-aware reductions, sum always returned the unrounded
   double accumulator; the value below is now the dtype-correct float32
   representation of that sum. */
$a = NumPower::array([[-156.50, 150.525435], [0, -39.151414]]);
print_r(NumPower::sum($a));
print_r(NumPower::sum($a, axis: 0)->toArray());
print_r(NumPower::sum($a, axis: 1)->toArray());
print_r(NumPower::sum($a[0]));
print_r(NumPower::sum([[0.12],[-0.513124]]));
?>
--EXPECT--
-45.1259765625Array
(
    [0] => -156.5
    [1] => 111.3740234375
)
Array
(
    [0] => -5.9745635986328
    [1] => -39.151412963867
)
-5.9745635986328-0.39312398433685