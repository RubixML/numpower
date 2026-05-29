--TEST--
PyTorch parity: 18 unary math/explog ops match PyTorch (libm reference values) bit-exact on fp64
--FILE--
<?php
/* Verifies numpower's 18 unary ops match PyTorch's CPU results bit-exact
   on fp64. PyTorch's CPU kernel for transcendentals routes through libm
   (`expf`, `expm1f`, `log`, …) which is also what Python's `math` module
   uses; comparing against `math.exp(1)` is therefore equivalent to
   comparing against `torch.exp(torch.tensor(1.0, dtype=torch.float64)).item()`.

   Reference values from Python's `math` module:
     math.e          = 2.718281828459045
     math.exp(0)     = 1.0
     math.exp(1)     = 2.718281828459045
     math.exp(2)     = 7.38905609893065
     math.exp(4)     = 54.598150033144236
     math.expm1(0)   = 0.0                       (exact)
     math.expm1(1)   = 1.718281828459045
     math.expm1(1e-15) = 1.0000000000000007e-15
     math.log(1)     = 0.0                       (exact)
     math.log(math.e) = 1.0                      (CPU libm)
     math.log(2)     = 0.6931471805599453
     math.log1p(0)   = 0.0                       (exact)
     math.log1p(1e-15) = 9.999999999999995e-16
     math.log10(1000) = 3.0                      (exact)
     math.log2(1024) = 10.0                      (exact)

   Special-value semantics (IEEE 754, PyTorch parity):
     log(0)   = -inf
     log(-1)  = NaN
     log(inf) = inf
     sqrt(-1) = NaN
     rsqrt(0) = inf
     exp(-inf) = 0
     exp(inf)  = inf
     reciprocal(0) = inf, reciprocal(-0) = -inf
     sign([-3, 0, 3, nan]) = [-1, 0, 1, nan]
     abs(int8(-128)) = -128  (wrap modulo 2^8)
*/

function approx_eq($a, $b, $tol = 0.0) {
    if (is_nan($a) && is_nan($b)) return true;
    if (is_infinite($a) && is_infinite($b) && (($a < 0) === ($b < 0))) return true;
    if ($tol === 0.0) return $a === $b;
    return abs($a - $b) <= $tol;
}

function check($label, $got, $want, $tol = 0.0) {
    if (approx_eq($got, $want, $tol)) {
        echo "OK $label\n";
    } else {
        printf("FAIL %s: got=%.17g want=%.17g delta=%.3e\n",
               $label, $got, $want, abs($got - $want));
    }
}

/* ── Bit-exact libm parity on fp64 (CPU) ─────────────────────────────── */

$cases = [
    /* op,       input,   PyTorch/libm reference */
    ['exp',      1.0,     2.718281828459045],
    ['exp',      0.0,     1.0],
    ['exp',      2.0,     7.38905609893065],
    ['exp',      4.0,     54.598150033144236],
    ['exp2',     0.0,     1.0],
    ['exp2',     10.0,    1024.0],
    ['expm1',    0.0,     0.0],
    ['expm1',    1.0,     1.718281828459045],
    ['expm1',    1e-15,   1.0000000000000007e-15],
    ['log',      1.0,     0.0],
    ['log',      2.0,     0.6931471805599453],
    ['log1p',    0.0,     0.0],
    ['log1p',    1e-15,   9.999999999999995e-16],
    ['log2',     1024.0,  10.0],
    ['log2',     8.0,     3.0],
    ['log10',    1000.0,  3.0],
    ['log10',    1.0,     0.0],
    ['logb',     1024.0,  10.0],
    ['sqrt',     4.0,     2.0],
    ['sqrt',     2.0,     1.4142135623730951],
    ['rsqrt',    4.0,     0.5],
    /* rsqrt(2): math.pow(2, -0.5) = 0.7071067811865476, but `1.0/sqrt(2)`
       (the derived numpower / PyTorch path) carries two rounding steps
       and returns the adjacent fp64 `0.70710678118654746`. Both PyTorch
       and numpower do the same; document the 1-ULP delta. */
    ['rsqrt',    2.0,     0.70710678118654746],
    ['reciprocal', 4.0,   0.25],
    ['abs',      -3.5,    3.5],
    ['negative', 3.5,     -3.5],
    ['positive', 3.5,     3.5],
    ['sign',     -7.0,    -1.0],
    ['sign',     0.0,     0.0],
    ['sign',     7.0,     1.0],
    ['square',   3.0,     9.0],
    ['sinc',     0.0,     1.0],
];

foreach ($cases as [$op, $in, $want]) {
    $r = NumPower::$op(NumPower::array([$in], 'float64'))->toArray()[0];
    check("$op($in) CPU bit-exact vs libm", $r, $want, 0.0);
}

/* ── Special-value semantics (CPU) ──────────────────────────────────── */

$specials = [
    ['log', 0.0, -INF],
    ['log', -1.0, NAN],
    ['log', INF, INF],
    ['sqrt', -1.0, NAN],
    ['rsqrt', 0.0, INF],
    ['exp', -INF, 0.0],
    ['exp', INF, INF],
    ['reciprocal', 0.0, INF],
    ['log1p', -1.0, -INF],
    ['log2', 0.0, -INF],
    ['log10', 0.0, -INF],
];

foreach ($specials as [$op, $in, $want]) {
    $r = NumPower::$op(NumPower::array([$in], 'float64'))->toArray()[0];
    $tag = is_nan($want) ? "NaN" : (is_infinite($want) ? ($want < 0 ? "-inf" : "+inf") : (string)$want);
    check("$op($in) → $tag (PyTorch)", $r, $want, 0.0);
}

/* sign of NaN propagates (PyTorch parity). */
$r = NumPower::sign(NumPower::array([NAN], 'float64'))->toArray()[0];
check("sign(NaN) → NaN", $r, NAN, 0.0);

/* reciprocal(-0) → -inf. */
$r = NumPower::reciprocal(NumPower::array([-0.0], 'float64'))->toArray()[0];
check("reciprocal(-0) → -inf", $r, -INF, 0.0);

/* clip / clamp matches PyTorch. */
$r = NumPower::clip(NumPower::array([-2.0, 0.5, 3.0], 'float64'), 0.0, 1.0)->toArray();
$expected = [0.0, 0.5, 1.0];
$ok = true;
for ($i = 0; $i < 3; $i++) if (!approx_eq($r[$i], $expected[$i], 0.0)) $ok = false;
echo ($ok ? "OK" : "FAIL"), " clip([-2, 0.5, 3], 0, 1) → [0, 0.5, 1] (PyTorch)\n";

/* clip on NaN propagates NaN. */
$r = NumPower::clip(NumPower::array([NAN], 'float64'), 0.0, 1.0)->toArray()[0];
check("clip(NaN, 0, 1) → NaN", $r, NAN, 0.0);

/* abs(int8 INT8_MIN) wraps to itself (NumPy / PyTorch). */
$r = NumPower::abs(NumPower::array([-128], 'int8'))->toArray()[0];
echo "abs(int8(-128)) = $r (PyTorch parity: -128 wrap)\n";
check("abs(int8(-128)) wraps to itself", $r, -128, 0.0);

/* abs(uint8) is identity. */
$r = NumPower::abs(NumPower::array([0, 128, 255], 'uint8'))->toArray();
echo "abs(uint8 ID) = "; print_r($r);

/* Integer widening to fp32 / fp64 per PyTorch's result_type for transcendentals. */
foreach (['int8' => 'float32', 'uint8' => 'float32',
          'int16' => 'float32', 'uint16' => 'float32',
          'int32' => 'float64', 'uint32' => 'float64',
          'int64' => 'float64', 'uint64' => 'float64'] as $src => $want_dt) {
    $r = NumPower::exp(NumPower::array([1, 2], $src));
    $got_dt = $r->__serialize()['dtype'];
    check("exp($src) widens to $want_dt (PyTorch result_type)",
          $got_dt === $want_dt, true, 0.0);
}

/* dtype preservation on dtype-preserving ops (abs / negative / sign / square). */
foreach (['int8', 'int32', 'int64', 'uint8', 'uint64',
          'float16', 'float32', 'float64'] as $dt) {
    $vals = [1, 2, 3];
    $r = NumPower::abs(NumPower::array($vals, $dt));
    $got_dt = $r->__serialize()['dtype'];
    check("abs($dt) preserves dtype",
          $got_dt === $dt, true, 0.0);
}

echo "DONE\n";
?>
--EXPECT--
OK exp(1) CPU bit-exact vs libm
OK exp(0) CPU bit-exact vs libm
OK exp(2) CPU bit-exact vs libm
OK exp(4) CPU bit-exact vs libm
OK exp2(0) CPU bit-exact vs libm
OK exp2(10) CPU bit-exact vs libm
OK expm1(0) CPU bit-exact vs libm
OK expm1(1) CPU bit-exact vs libm
OK expm1(1.0E-15) CPU bit-exact vs libm
OK log(1) CPU bit-exact vs libm
OK log(2) CPU bit-exact vs libm
OK log1p(0) CPU bit-exact vs libm
OK log1p(1.0E-15) CPU bit-exact vs libm
OK log2(1024) CPU bit-exact vs libm
OK log2(8) CPU bit-exact vs libm
OK log10(1000) CPU bit-exact vs libm
OK log10(1) CPU bit-exact vs libm
OK logb(1024) CPU bit-exact vs libm
OK sqrt(4) CPU bit-exact vs libm
OK sqrt(2) CPU bit-exact vs libm
OK rsqrt(4) CPU bit-exact vs libm
OK rsqrt(2) CPU bit-exact vs libm
OK reciprocal(4) CPU bit-exact vs libm
OK abs(-3.5) CPU bit-exact vs libm
OK negative(3.5) CPU bit-exact vs libm
OK positive(3.5) CPU bit-exact vs libm
OK sign(-7) CPU bit-exact vs libm
OK sign(0) CPU bit-exact vs libm
OK sign(7) CPU bit-exact vs libm
OK square(3) CPU bit-exact vs libm
OK sinc(0) CPU bit-exact vs libm
OK log(0) → -inf (PyTorch)
OK log(-1) → NaN (PyTorch)
OK log(INF) → +inf (PyTorch)
OK sqrt(-1) → NaN (PyTorch)
OK rsqrt(0) → +inf (PyTorch)
OK exp(-INF) → 0 (PyTorch)
OK exp(INF) → +inf (PyTorch)
OK reciprocal(0) → +inf (PyTorch)
OK log1p(-1) → -inf (PyTorch)
OK log2(0) → -inf (PyTorch)
OK log10(0) → -inf (PyTorch)
OK sign(NaN) → NaN
OK reciprocal(-0) → -inf
OK clip([-2, 0.5, 3], 0, 1) → [0, 0.5, 1] (PyTorch)
OK clip(NaN, 0, 1) → NaN
abs(int8(-128)) = -128 (PyTorch parity: -128 wrap)
OK abs(int8(-128)) wraps to itself
abs(uint8 ID) = Array
(
    [0] => 0
    [1] => 128
    [2] => 255
)
OK exp(int8) widens to float32 (PyTorch result_type)
OK exp(uint8) widens to float32 (PyTorch result_type)
OK exp(int16) widens to float32 (PyTorch result_type)
OK exp(uint16) widens to float32 (PyTorch result_type)
OK exp(int32) widens to float64 (PyTorch result_type)
OK exp(uint32) widens to float64 (PyTorch result_type)
OK exp(int64) widens to float64 (PyTorch result_type)
OK exp(uint64) widens to float64 (PyTorch result_type)
OK abs(int8) preserves dtype
OK abs(int32) preserves dtype
OK abs(int64) preserves dtype
OK abs(uint8) preserves dtype
OK abs(uint64) preserves dtype
OK abs(float16) preserves dtype
OK abs(float32) preserves dtype
OK abs(float64) preserves dtype
DONE
