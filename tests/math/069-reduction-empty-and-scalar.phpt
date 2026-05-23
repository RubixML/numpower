--TEST--
NumPower reductions on 0-D scalars and edge cases
--FILE--
<?php
/* The Reduce_* helpers special-case n=0 to match NumPy:
     sum identity = 0
     prod identity = 1
     min/max on empty throw
     mean on empty returns NaN
   0-D scalars have n=1 and the reduction is the scalar itself. */

/* 0-D scalar: trivial reduction. */
$s = new NDArray(42, 'float32');
$ok_scalar = NumPower::sum($s)  === 42.0
          && NumPower::prod($s) === 42.0
          && NumPower::min($s)  === 42.0
          && NumPower::max($s)  === 42.0
          && NumPower::mean($s) === 42.0;
echo "0-D scalar: ", $ok_scalar ? "ok\n" : "FAIL\n";

/* NaN propagation. */
$nan = new NDArray([1.0, 2.0, NAN, 4.0], 'float64');
echo "sum NaN: ",  is_nan(NumPower::sum($nan))  ? "ok\n" : "FAIL\n";
echo "prod NaN: ", is_nan(NumPower::prod($nan)) ? "ok\n" : "FAIL\n";
echo "max NaN: ",  is_nan(NumPower::max($nan))  ? "ok\n" : "FAIL\n";

/* +INF / -INF semantics. */
$inf = new NDArray([1.0, 2.0, INF, 4.0], 'float64');
echo "sum INF: ",  NumPower::sum($inf) === INF ? "ok\n" : "FAIL\n";
echo "max INF: ",  NumPower::max($inf) === INF ? "ok\n" : "FAIL\n";

$ninf = new NDArray([1.0, 2.0, -INF, 4.0], 'float64');
echo "min -INF: ", NumPower::min($ninf) === -INF ? "ok\n" : "FAIL\n";
?>
--EXPECT--
0-D scalar: ok
sum NaN: ok
prod NaN: ok
max NaN: ok
sum INF: ok
max INF: ok
min -INF: ok
