--TEST--
Arithmetic ops accept string scalars and preserve precision for float128 / int64 / uint64
--FILE--
<?php
/* Three regressions covered:
   1. ZVAL_TO_NDARRAY used to reject IS_STRING outright — fp128 and uint64
      callers had no precision-loss-free way to pass a scalar operand.
   2. compute_dtype_for_arithmetic routed int64 / uint64 through float64,
      so any value past 2^53 silently rounded.
   3. The operator-overload path (`$arr + $string`) silently fell through
      to PHP's default `do_operation` and produced garbage.
   Now every binary op handles IS_STRING via NDArray_EncodeZvalToDtype and
   int64 / uint64 compute natively. */

/* float128 scalar via string — full-precision parse */
$f = new NDArray(['1e30', '2e30'], 'float128');
$r = NumPower::add($f, '5e30');
echo "f128 add string=", (string)$r[0], "\n";      // 6e30
echo "f128 add string=", (string)$r[1], "\n";      // 7e30

/* uint64 scalar via string — full-precision parse */
$u = new NDArray(['18446744073709551610'], 'uint64');
$r = NumPower::add($u, '5');
echo "u64+5='", (string)$r[0], "'\n";              // 18446744073709551615 (UINT64_MAX)

/* int64 scalar via string — wraps at INT64_MAX (PyTorch parity) */
$i = new NDArray(['9223372036854775806'], 'int64');
$r = NumPower::add($i, '1');
echo "i64 add: '", (string)$r[0], "'\n";            // 9223372036854775807

/* Operator overload also accepts the precision-preserving path. */
$u = new NDArray(['18446744073709551600'], 'uint64');
$r = $u + 10;
echo "u64 op +10: '", (string)$r[0], "'\n";         // 18446744073709551610

/* Same for subtract / multiply / mod / pow on int64 */
$i = new NDArray(['1000000000000000000'], 'int64');
$r = NumPower::subtract($i, '500000000000000000');
echo "i64 sub: '", (string)$r[0], "'\n";            // 500000000000000000

$r = NumPower::multiply(new NDArray([1000, 1000], 'int64'), '1000000000');
echo "i64 mul: '", (string)$r[0], "'\n";            // 1000000000000

$r = NumPower::mod(new NDArray([10000000000], 'int64'), '7');
echo "i64 mod: '", (string)$r[0], "'\n";            // 10000000000 mod 7 = 4

$r = NumPower::pow(new NDArray([10], 'int64'), '18');
echo "i64 pow: '", (string)$r[0], "'\n";            // 10^18 = 1000000000000000000

/* String scalar adopts the peer's dtype (PyTorch weak-scalar rule). */
$f32 = new NDArray([1.5, 2.5], 'float32');
$r = NumPower::add($f32, '1.0');                    /* stays float32 */
echo "f32+str: type=", gettype($r[0]), " val=", (string)$r[0], "\n";

/* A bare string with no peer NDArray throws (we cannot infer dtype). */
try {
    NumPower::add('1.5', '2.5');
    echo "BUG: bare strings accepted\n";
} catch (\Error $e) {
    echo "bare strings: ", $e->getMessage(), "\n";
}

/* Boolean scalar promotes too (NDArray_EncodeZvalToDtype handles IS_TRUE / IS_FALSE). */
$ones = new NDArray([1, 2, 3], 'int32');
$r = NumPower::add($ones, true);
echo "i32+true: ", (string)$r[0], " ", (string)$r[1], " ", (string)$r[2], "\n";
$r = NumPower::multiply($ones, false);
echo "i32*false: ", (string)$r[0], " ", (string)$r[1], " ", (string)$r[2], "\n";
?>
--EXPECT--
f128 add string=6000000000000000000000000000000
f128 add string=7000000000000000000000000000000
u64+5='18446744073709551615'
i64 add: '9223372036854775807'
u64 op +10: '18446744073709551610'
i64 sub: '500000000000000000'
i64 mul: '1000000000000'
i64 mod: '4'
i64 pow: '1000000000000000000'
f32+str: type=double val=2.5
bare strings: Cannot infer dtype for a string scalar without an NDArray peer.
i32+true: 2 3 4
i32*false: 0 0 0
