--TEST--
Arithmetic at integer-dtype boundaries: overflow wraps (matches PyTorch's defined behavior on x86-64 GCC)
--FILE--
<?php
/* PyTorch keeps the result dtype and lets overflow wrap via two's-complement
   on signed ints / modulo on unsigned ints. Our cast-back through
   ndarray_set_from_double matches this on x86-64 GCC (where implementation-
   defined behavior is wrap, not saturate). */

/* int8 wraps: 127 + 1 = -128 */
$a = NumPower::array([127, 127, 127, 127], 'int8');
$r = $a + 1;
echo "int8 127+1: ", json_encode($r->toArray()), "\n";

/* int8 multiply wraps: 100*2 = 200 → -56 */
$a = NumPower::array([100, 100, 100, 100], 'int8');
$r = $a * 2;
echo "int8 100*2: ", json_encode($r->toArray()), "\n";

/* uint8 wraps modulo 256: 255 + 1 = 0 */
$a = NumPower::array([255, 255, 255, 255], 'uint8');
$r = $a + 1;
echo "uint8 255+1: ", json_encode($r->toArray()), "\n";

/* int32 boundary: 2^31 - 1 + 0 = max int32 */
$a = NumPower::array([2147483647, 2147483647, 2147483647, 2147483647], 'int32');
$r = $a + 0;
echo "int32 max: ", json_encode($r->toArray()), "\n";

/* int16 negative boundary */
$a = NumPower::array([-32768, -32768, -32768, -32768], 'int16');
$r = $a + 0;
echo "int16 min: ", json_encode($r->toArray()), "\n";

/* Float subnormal and INF */
$a = NumPower::array([INF, INF, -INF, -INF], 'float64');
$b = NumPower::array([1.0, 0.0, 1.0, 0.0], 'float64');
$r = $a + $b;
$out = $r->toArray();
echo "INF + finite/zero: ",
     ($out[0]===INF ? 'INF' : (string)$out[0]),
     ',', ($out[1]===INF ? 'INF' : (string)$out[1]),
     ',', ($out[2]===-INF ? '-INF' : (string)$out[2]),
     ',', ($out[3]===-INF ? '-INF' : (string)$out[3]), "\n";

/* NaN propagation */
$a = NumPower::array([NAN, NAN, NAN, NAN], 'float64');
$r = $a + 1;
foreach ($r->toArray() as $v) {
    if (!is_nan($v)) { echo "FAIL: NaN not propagated\n"; break; }
}
echo "NaN propagation: ok\n";

/* float division by zero gives INF / NaN */
$a = NumPower::array([1.0, 0.0, -1.0], 'float64');
$b = NumPower::array([0.0, 0.0, 0.0], 'float64');
$r = $a / $b;
$out = $r->toArray();
$s = [];
foreach ($out as $v) {
    if (is_nan($v))      $s[] = 'NaN';
    else if ($v === INF)  $s[] = 'INF';
    else if ($v === -INF) $s[] = '-INF';
    else                  $s[] = (string)$v;
}
echo "float / 0: ", implode(',', $s), "\n";
?>
--EXPECT--
int8 127+1: [-128,-128,-128,-128]
int8 100*2: [-56,-56,-56,-56]
uint8 255+1: [0,0,0,0]
int32 max: [2147483647,2147483647,2147483647,2147483647]
int16 min: [-32768,-32768,-32768,-32768]
INF + finite/zero: INF,INF,-INF,-INF
NaN propagation: ok
float / 0: INF,NaN,-INF
