--TEST--
NumPower fp128 sqrt/sin uses libquadmath full precision (regression guard for HAVE_QUADMATH include-order bug)
--FILE--
<?php
/* Regression guard: `NDARRAY_FP128_SQRT` / `_SIN` previously expanded
   inline in `ndarray_types.h` behind `#if HAVE_QUADMATH`. Because
   `ndarray_types.h` is included via the project's header chain before
   any `.c` file gets to include `config.h`, the macro silently froze
   at the `sqrtl`/`sinl` long-double fallback (~64-bit precision)
   instead of `sqrtq`/`sinq` (113-bit precision).
   The fix routes through out-of-line `ndarray_fp128_sqrt` /
   `ndarray_fp128_sin` defined in `ndarray_types.c` where `config.h`
   is processed first. This test verifies the libquadmath path is
   live by checking that fp128 sqrt preserves digits beyond the
   ~18-digit long-double bound. */

function preserved_digits($got, $want) {
    /* Count leading matching digits (ignoring sign / decimal point). */
    $g = str_replace(['-', '.', '+'], '', (string)$got);
    $w = str_replace(['-', '.', '+'], '', (string)$want);
    $n = min(strlen($g), strlen($w));
    for ($i = 0; $i < $n; $i++) {
        if ($g[$i] !== $w[$i]) return $i;
    }
    return $n;
}

/* sqrt(2) in 34-digit fp128:
   1.41421356237309504880168872420969807856967187537694... */
$two = NumPower::array(['2'], 'float128');
$s   = NumPower::sqrt($two)->toArray()[0];
$exp = '1.41421356237309504880168872420969807';
$digits = preserved_digits($s, $exp);
echo "sqrt(2) preserved digits: $digits\n";  /* >= 30 means libquadmath is live */
if ($digits >= 30) echo "OK fp128 sqrt uses libquadmath precision\n";
else                echo "FAIL fp128 sqrt only preserved $digits digits (got $s)\n";

/* Probe: detect libquadmath presence by feeding a 33-digit literal that
   only true fp128 storage round-trips (DD canonicalizes ~32 digits). */
$probe = (string)(new NDArray(['9.999999999999999999999999999999999e-10'], 'float128'));
$is_quadmath = strpos($probe, '9.9999999999999999999999999999999') !== false;

/* sinc(1/6) identity: sin(π·x)/(π·x) at x = 1/6 evaluates to
   sin(π/6) / (π/6) = 0.5 / (π/6) = 3/π. We use sinc rather than
   calling sin directly because the unary dispatcher exposes sinc;
   any precision loss in the underlying sin would still surface here. */
$sinc_in = NumPower::array(['0.1666666666666666666666666666666667'], 'float128');
$out     = NumPower::sinc($sinc_in)->toArray()[0];
/* 3/π ≈ 0.954929658551372014613302580235082... */
$want    = '0.9549296585513720146133025802350917';
$digits  = preserved_digits($out, $want);
echo "sinc(1/6) preserved digits: $digits\n";
/* libquadmath path uses M_PIq + sinq + DD div, preserving ~30 digits.
   DD-emulation path falls back to fp64 sin and only preserves ~15 digits
   (fp64 mantissa) — both are healthy floors for the respective build. */
$threshold = $is_quadmath ? 25 : 14;
if ($digits >= $threshold) echo "OK fp128 sinc uses high-precision pi+sin\n";
else                       echo "WARN fp128 sinc only preserved $digits digits (got $out)\n";

/* CPU rsqrt(0) on fp128 returns +Inf (the dd_div zero check), GPU returns NaN
   (no zero check in CUDA dd_div). Document this divergence — the test
   doesn't fail either way, but the difference is observable. */
$rsqrt0 = NumPower::rsqrt(NumPower::array(['0', '4'], 'float128'))->toArray();
echo "fp128 rsqrt([0,4]) CPU [1] = ", $rsqrt0[1], "\n";

/* clip validation: malformed strings should raise an exception now
   instead of silently parsing to 0. */
try {
    NumPower::clip(NumPower::array([1.0], 'float64'), 'twelve', '5');
    echo "FAIL clip accepted garbage 'twelve'\n";
} catch (\Error $e) {
    echo "OK clip rejects garbage: ", $e->getMessage(), "\n";
}

try {
    NumPower::clip(NumPower::array([1.0], 'float64'), '1.5e', '5');
    echo "FAIL clip accepted malformed exponent '1.5e'\n";
} catch (\Error $e) {
    echo "OK clip rejects malformed exponent: ", $e->getMessage(), "\n";
}

/* Strengthen rsqrt-was-arccos regression: arccos(0.25) ≈ 1.318 != rsqrt(0.25) = 2.0. */
$r = NumPower::rsqrt(NumPower::array([0.25], 'float32'))->toArray()[0];
if (abs($r - 2.0) < 1e-5) echo "OK rsqrt(0.25) == 2.0 (not arccos(0.25) ≈ 1.318)\n";
else                       echo "FAIL rsqrt(0.25) = $r (expected 2.0)\n";

echo "DONE\n";
?>
--EXPECTF--
sqrt(2) preserved digits: %d
OK fp128 sqrt uses libquadmath precision
sinc(1/6) preserved digits: %d
OK fp128 sinc uses high-precision pi+sin
fp128 rsqrt([0,4]) CPU [1] = 0.5
OK clip rejects garbage: %s
OK clip rejects malformed exponent: %s
OK rsqrt(0.25) == 2.0 (not arccos(0.25) ≈ 1.318)
DONE
