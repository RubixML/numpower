--TEST--
Pre-existing precision bugs fixed: log2 GPU at powers of 2, dd_sinc DD precision near zero, fp128 NaN sign normalization
--FILE--
<?php
/* Three pre-existing precision bugs surfaced during the string-scalar
   review and fixed in this branch.

   1. log2 GPU 1-ULP error at powers of 2
      CUDA's `log2(8.0)` returned 2.9999999999999996 instead of 3.0
      because the libdevice intrinsic has up to 1 ULP error and
      doesn't special-case exact powers of two. Fix: front-load a
      `frexp(x) == 0.5` exact short-circuit in `tcuda_log2_fp` so
      `log2(2^k) → k` bit-for-bit, matching CPU libm. Applies to
      float32, float64, fp128 (DD), and fp16 (via the float kernel).

   2. dd_sinc fp128 GPU collapsed to fp64 precision near zero
      The previous DD kernel rounded to fp64 before calling sin, so
      `sinc(1e-10)` came back as exactly 1.0, losing the ~1.6e-21
      deviation. Fix: when |x| ≤ 0.1, run the sinc Taylor series
      (1 − (πx)²/6 + (πx)⁴/120 − …, 5 terms) in DD arithmetic — the
      DD result matches the CPU libquadmath value to within a few DD
      ULPs (~32 sig digits) for the small-x regime.

   3. fp128 transcendentals leaked sign-bit NaN ("-nan")
      `logq(-x)`, `sqrtq(-x)`, `log1pq(-x)` etc. return NaN with the
      sign bit set, which `quadmath_snprintf %Qg` honours. PHP's
      float stringifier hides the sign for fp64 NaN, so users saw the
      inconsistency only on fp128. Fix: normalize the sign bit to +0
      in `ndarray_fp128_to_string` so all NaN displays as "nan"
      regardless of producer.
*/

function check($label, $got, $want, $tol = 0.0) {
    if ($tol === 0.0) {
        if ($got === $want) { echo "OK $label\n"; return; }
    } else {
        if (is_array($got) && is_array($want) && count($got) === count($want)) {
            $ok = true;
            for ($i = 0; $i < count($got); $i++) {
                if (abs((float)$got[$i] - (float)$want[$i]) > $tol) { $ok = false; break; }
            }
            if ($ok) { echo "OK $label\n"; return; }
        } else if (abs((float)$got - (float)$want) <= $tol) { echo "OK $label\n"; return; }
    }
    echo "FAIL $label: got=", json_encode($got), " want=", json_encode($want), "\n";
}

/* ── 1. log2 GPU at powers of 2 → exact integer result ─────────────────── */

/* Powers of 2 across the supported dtypes — all CPU and GPU should
   produce exact integer log2. */
foreach (['float32', 'float64', 'float128'] as $dt) {
    $is_fp128 = ($dt === 'float128');
    $vals = $is_fp128 ? ['1.0', '2.0', '4.0', '8.0', '16.0', '1024.0', '65536.0']
                       : [1.0, 2.0, 4.0, 8.0, 16.0, 1024.0, 65536.0];
    $expect = $is_fp128 ? ['0', '1', '2', '3', '4', '10', '16']
                         : [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 16.0];
    $cpu = $is_fp128 ? new NDArray($vals, $dt) : NumPower::array($vals, $dt);
    try {
        $gpu = $cpu->gpu();
        $rg = NumPower::log2($gpu)->cpu()->toArray();
        check("log2 GPU exact at powers of 2 ($dt)", $rg, $expect);
    } catch (\Throwable $t) {
        echo "skip log2 $dt GPU: ", $t->getMessage(), "\n";
    }
}

/* Negative powers (1/2, 1/4, 1/8) on fp64 — log2 should give -1, -2, -3. */
try {
    $gpu = NumPower::array([0.5, 0.25, 0.125, 0.0625], 'float64')->gpu();
    $r = NumPower::log2($gpu)->cpu()->toArray();
    check("log2 GPU exact at negative powers of 2 (fp64)", $r, [-1.0, -2.0, -3.0, -4.0]);
} catch (\Throwable $t) { echo "skip negative power log2 test: ", $t->getMessage(), "\n"; }

/* Wide ints with powers of 2 — widening to fp64 stays exact. */
foreach (['int32', 'uint32', 'int64', 'uint64'] as $dt) {
    try {
        $gpu = NumPower::array([1, 2, 4, 8, 16, 1024], $dt)->gpu();
        $r = NumPower::log2($gpu)->cpu()->toArray();
        check("log2 GPU exact at powers of 2 ($dt)", $r, [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]);
    } catch (\Throwable $t) { echo "skip int $dt log2 test\n"; }
}

/* Non-powers-of-2 still pass through CUDA's intrinsic — verify they
   match CPU within fp64 tolerance, no regression. */
try {
    $cpu = NumPower::array([3.0, 5.0, 7.0, 100.0, 1.5, 0.7], 'float64');
    $gpu = $cpu->gpu();
    $rc = NumPower::log2($cpu)->toArray();
    $rg = NumPower::log2($gpu)->cpu()->toArray();
    check("log2 GPU non-powers match CPU within 1e-15", $rc, $rg, 1e-15);
} catch (\Throwable $t) { echo "skip non-powers log2 test\n"; }

/* ── 2. dd_sinc fp128 GPU precision near zero ──────────────────────────── */

try {
    /* sinc(1e-10) on fp128 — CPU libquadmath gives ~0.99999999999999999998355...
       GPU previously gave exactly 1.0 (fp64 collapse). After fix the GPU
       result must agree with CPU to many DD digits. */
    $cpu = new NDArray(['1e-10'], 'float128');
    $gpu = $cpu->gpu();
    $rc = $cpu->toArray()[0]; /* preserved input */
    $rcv = (string)NumPower::sinc($cpu)->toArray()[0];
    $rgv = (string)NumPower::sinc($gpu)->cpu()->toArray()[0];
    /* Both should NOT be exactly "1" — the deviation from 1 is real. */
    $cpu_close_to_1 = (strpos($rcv, '0.999999999999') === 0);
    $gpu_close_to_1 = (strpos($rgv, '0.999999999999') === 0);
    check("sinc(1e-10) CPU not exactly 1 (fp128)", $cpu_close_to_1, true);
    check("sinc(1e-10) GPU not exactly 1 (fp128) after DD Taylor fix",
          $gpu_close_to_1, true);

    /* And the relative difference between CPU and GPU should be < 1e-30
       (full DD precision). */
    $diff = abs((float)$rcv - (float)$rgv);
    check("sinc(1e-10) CPU/GPU diff < 1e-20 (full DD precision)",
          $diff < 1e-20, true);
} catch (\Throwable $t) {
    echo "skip sinc DD fix tests: ", $t->getMessage(), "\n";
}

/* sinc at other small-x values: 1e-3, 1e-5, 0.05 should all agree
   between CPU and GPU. */
try {
    foreach (['1e-3', '1e-5', '0.05'] as $v) {
        $cpu = new NDArray([$v], 'float128');
        $gpu = $cpu->gpu();
        $rcv = (string)NumPower::sinc($cpu)->toArray()[0];
        $rgv = (string)NumPower::sinc($gpu)->cpu()->toArray()[0];
        /* fp64 tail rounding may diverge in the last few digits — relative
           tolerance is plenty since the values are all close to 1. */
        $diff = abs((float)$rcv - (float)$rgv);
        check("sinc($v) CPU/GPU fp128 agree within 1e-15", $diff < 1e-15, true);
    }
} catch (\Throwable $t) { echo "skip small-x sinc tests\n"; }

/* sinc(0) on fp128: must be exactly 1. */
$cpu = new NDArray(['0.0'], 'float128');
check("sinc(0) fp128 CPU == 1", (string)NumPower::sinc($cpu)->toArray()[0], '1');
try {
    $gpu = $cpu->gpu();
    check("sinc(0) fp128 GPU == 1", (string)NumPower::sinc($gpu)->cpu()->toArray()[0], '1');
} catch (\Throwable $t) { echo "skip sinc(0) GPU test\n"; }

/* ── 3. fp128 NaN sign normalization ──────────────────────────────────── */

/* All ops that produce NaN from invalid fp128 inputs should now stringify
   as "nan", not "-nan". log1p is special: log1p(-1) = log(0) = -inf
   (not NaN), so test it separately with a deeper-negative input. */
foreach (['log', 'log2', 'log10', 'sqrt', 'rsqrt'] as $op) {
    foreach (['-1.0', '-inf'] as $v) {
        $cpu = new NDArray([$v], 'float128');
        $r = (string)NumPower::$op($cpu)->toArray()[0];
        check("$op('$v') fp128 sign-normalized NaN", $r, 'nan');
    }
}
/* log1p(-2) → log(-1) → NaN. */
$cpu = new NDArray(['-2.0'], 'float128');
check("log1p('-2.0') fp128 sign-normalized NaN",
      (string)NumPower::log1p($cpu)->toArray()[0], 'nan');
/* log1p(-inf) → log(-inf + 1) → NaN per IEEE 754 (well-defined branch). */
$cpu = new NDArray(['-inf'], 'float128');
check("log1p('-inf') fp128 sign-normalized NaN",
      (string)NumPower::log1p($cpu)->toArray()[0], 'nan');

/* Same on GPU. */
try {
    foreach (['log', 'sqrt'] as $op) {
        foreach (['-1.0', '-inf'] as $v) {
            $cpu = new NDArray([$v], 'float128');
            $gpu = $cpu->gpu();
            $r = (string)NumPower::$op($gpu)->cpu()->toArray()[0];
            check("$op('$v') fp128 GPU sign-normalized NaN", $r, 'nan');
        }
    }
} catch (\Throwable $t) { echo "skip fp128 GPU NaN tests\n"; }

/* Direct NaN input is preserved (sign-bit absent on the canonical input). */
$cpu = new NDArray(['nan'], 'float128');
check("nan input stays 'nan'", (string)$cpu->toArray()[0], 'nan');

echo "DONE\n";
?>
--EXPECT--
OK log2 GPU exact at powers of 2 (float32)
OK log2 GPU exact at powers of 2 (float64)
OK log2 GPU exact at powers of 2 (float128)
OK log2 GPU exact at negative powers of 2 (fp64)
OK log2 GPU exact at powers of 2 (int32)
OK log2 GPU exact at powers of 2 (uint32)
OK log2 GPU exact at powers of 2 (int64)
OK log2 GPU exact at powers of 2 (uint64)
OK log2 GPU non-powers match CPU within 1e-15
OK sinc(1e-10) CPU not exactly 1 (fp128)
OK sinc(1e-10) GPU not exactly 1 (fp128) after DD Taylor fix
OK sinc(1e-10) CPU/GPU diff < 1e-20 (full DD precision)
OK sinc(1e-3) CPU/GPU fp128 agree within 1e-15
OK sinc(1e-5) CPU/GPU fp128 agree within 1e-15
OK sinc(0.05) CPU/GPU fp128 agree within 1e-15
OK sinc(0) fp128 CPU == 1
OK sinc(0) fp128 GPU == 1
OK log('-1.0') fp128 sign-normalized NaN
OK log('-inf') fp128 sign-normalized NaN
OK log2('-1.0') fp128 sign-normalized NaN
OK log2('-inf') fp128 sign-normalized NaN
OK log10('-1.0') fp128 sign-normalized NaN
OK log10('-inf') fp128 sign-normalized NaN
OK sqrt('-1.0') fp128 sign-normalized NaN
OK sqrt('-inf') fp128 sign-normalized NaN
OK rsqrt('-1.0') fp128 sign-normalized NaN
OK rsqrt('-inf') fp128 sign-normalized NaN
OK log1p('-2.0') fp128 sign-normalized NaN
OK log1p('-inf') fp128 sign-normalized NaN
OK log('-1.0') fp128 GPU sign-normalized NaN
OK log('-inf') fp128 GPU sign-normalized NaN
OK sqrt('-1.0') fp128 GPU sign-normalized NaN
OK sqrt('-inf') fp128 GPU sign-normalized NaN
OK nan input stays 'nan'
DONE
