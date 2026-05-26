--TEST--
NumPower::sum / NumPower::prod widen narrow ints and use native wide-precision accumulators
--FILE--
<?php
/* NumPy / PyTorch convention: sum and prod widen narrow integer dtypes
   to int64 / uint64 so the accumulator can't overflow at the dtype's
   range. The CPU implementation also uses native int64_t / uint64_t /
   ndarray_fp128_t accumulators for wide-precision results (the prior
   double accumulator silently rounded past 2^53). */

/* PHP type after collapsing a 0-D reduction to a dtype-correct scalar:
   - string for float128 / uint64,
   - int for the remaining integer dtypes,
   - float for the remaining floats. */
function phptype($v) {
    if (is_int($v))    return 'int';
    if (is_float($v))  return 'float';
    if (is_string($v)) return 'string';
    return gettype($v);
}

/* Cross-platform fp128 stringification: libquadmath emits full decimal
   form, DD-emulation (macOS/MSVC) emits scientific notation for large
   magnitudes. Funnel through `sprintf('%g', (float)...)` for a portable
   "<mantissa>e<exp>" form; the values exercised here are exactly
   representable in fp64 so the (float) cast is loss-free for assertion
   purposes. */
function fp128_norm($v) {
    $s = (string)$v;
    if ($s === '0' || $s === '0.0') return '0';
    return sprintf('%g', (float)$s);
}

/* Narrow int widening: int8 prod = 5*5*5*5 = 625 (overflows int8 = -127..127).
   With widening to int64 the result is exact. */
$i = new NDArray([5, 5, 5, 5], 'int8');
$p = NumPower::prod($i);
echo "i8 prod=$p type=", phptype($p), "\n";   /* expects 625, int */

/* Unsigned narrow widens to uint64 — collapsed scalar returns a string
   (uint64 may exceed PHP_INT_MAX, so the encoding is dtype-aware string). */
$u = new NDArray([200, 200, 200], 'uint8');
$s = NumPower::sum($u);
echo "u8 sum=$s type=", phptype($s), "\n";    /* 600, string */

/* uint64 sum past 2^53 — native accumulator keeps every bit. */
$u = new NDArray([(string)PHP_INT_MAX, '1', '1', '1', '1'], 'uint64');
echo "u64 sum=", NumPower::sum($u), "\n";     /* 9223372036854775811 (string) */

/* int64 sum near INT64_MAX — int64 accumulator wraps per C semantics. */
$i = new NDArray(['9223372036854775806', '1'], 'int64');
echo "i64 sum=", NumPower::sum($i), "\n";     /* 9223372036854775807 (INT64_MAX) */

/* float128 sum — native fp128 accumulator (DD on macOS / non-GCC). */
$f = new NDArray(['1e30', '2e30', '3e30'], 'float128');
echo "f128 sum=", fp128_norm(NumPower::sum($f)), "\n";    /* 6e30 in fp128 precision */
echo "f128 prod=", NumPower::prod(new NDArray(['1.5', '2', '3'], 'float128')), "\n"; /* 9 */

/* Narrow floats widen to float32 so the sum doesn't saturate at the
   dtype's tiny range (float4 max = 6, float8 E4M3 max = 240). */
$q = new NDArray([1, 2, 3, 4, 4], 'float4');   /* 5 quantises to 4 in float4 */
$s = NumPower::sum($q);
echo "f4 sum=$s type=", phptype($s), "\n";    /* 14, float (not 6) */

/* float8 E4M3 values around 100 quantise to 96 or 104 (step = 8 in that
   binade); use exact-representable values to keep the assertion stable. */
$q = new NDArray([96, 96, 96], 'float8');
$s = NumPower::sum($q);
echo "f8 sum=$s\n";                            /* 288 — no saturation */

/* Axis reduction also widens. */
$m = NumPower::array([[1, 2, 3], [4, 5, 6]], 'int8');
$r = NumPower::sum($m, 0);
echo "i8 sum axis=0 ", json_encode($r->toArray()), "\n"; /* [5, 7, 9] */
$r = NumPower::sum($m, 1);
echo "i8 sum axis=1 ", json_encode($r->toArray()), "\n"; /* [6, 15] */

/* Negative axis (numpy semantics). */
$r = NumPower::sum($m, -1);
echo "i8 sum axis=-1 ", json_encode($r->toArray()), "\n"; /* [6, 15] */

/* uint64 axis reduction past 2^53 — native accumulator. */
$u = NumPower::array([['18446744073709551610', '1'], ['1', '2']], 'uint64');
$r = NumPower::sum($u, 0);
echo "u64 sum axis=0 ", json_encode($r->toArray()), "\n"; /* ["18446744073709551611", "3"] */

/* Explicit null axis (per stub `?int $axis = null`) does the same as
   the no-second-argument call — no deprecation warning. */
$set = error_reporting(E_ALL);
echo "explicit null axis = ", NumPower::sum(NumPower::array([1.0, 2.0, 3.0]), null), "\n";
error_reporting($set);

/* Empty axis = identity-fill (0 for sum, 1 for prod). */
$empty = NumPower::zeros([0, 3], 'int32');
$r = NumPower::sum($empty, 0);
echo "empty axis=0 sum ", json_encode($r->toArray()), "\n"; /* [0,0,0] */
$r = NumPower::prod($empty, 0);
echo "empty axis=0 prod ", json_encode($r->toArray()), "\n"; /* [1,1,1] */

/* 1-D input with axis=0 collapses to dtype-correct scalar. */
$v = NumPower::array([1, 2, 3], 'int32');
$r = NumPower::sum($v, 0);
echo "1d axis=0: $r type=", phptype($r), "\n"; /* 6, int */

/* Out-of-range axis throws. */
try {
    NumPower::sum($m, 5);
    echo "BUG: out-of-range axis accepted\n";
} catch (\Error $e) {
    echo "axis-oob: caught\n";
}
?>
--EXPECT--
i8 prod=625 type=int
u8 sum=600 type=string
u64 sum=9223372036854775811
i64 sum=9223372036854775807
f128 sum=6.0e+30
f128 prod=9
f4 sum=14 type=float
f8 sum=288
i8 sum axis=0 [5,7,9]
i8 sum axis=1 [6,15]
i8 sum axis=-1 [6,15]
u64 sum axis=0 ["18446744073709551611","3"]
explicit null axis = 6
empty axis=0 sum [0,0,0]
empty axis=0 prod [1,1,1]
1d axis=0: 6 type=int
axis-oob: caught
