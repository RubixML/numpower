--TEST--
fp128 GPU DD-precision transcendentals: exp/log family now match CPU libquadmath to ~32 decimal digits
--SKIPIF--
<?php
try {
    $a = NumPower::array([1.0])->gpu();
    if (!$a->isGPU()) die("skip GPU not available");
} catch (Throwable $t) {
    die("skip GPU not available: " . $t->getMessage());
}
?>
--FILE--
<?php
/* Before this branch, GPU fp128 transcendentals (exp / log / expm1 /
   log1p / exp2 / log2 / log10) routed through fp64 libm intrinsics:
   `dd_make(exp(dd_to_double(x)), 0.0)` collapsed the result to ~15
   sig digits regardless of how many DD bits the host packed in.
   This branch replaces the per-op DD kernel body with a full
   DD-arithmetic implementation:
     - exp: range reduction (k·ln2 + r) + 20-term Horner of Taylor on r
     - log: frexp + atanh substitution + 26-term Horner of atanh series
     - expm1 / log1p: Taylor for small |x|, dd_exp / dd_log otherwise
     - exp2 / log2 / log10: derived from exp / log via DD constants

   Two non-obvious fixes were needed for the GPU port:
   1. `dd_two_prod` was using `double p = a * b; double e = fma(a, b, -p)`
      where NVCC's default `-fmad=true` silently contracts `a * b` into
      a fused multiply pattern that gives a 1-ULP-off rounding, breaking
      the two_prod invariant. Forcing `__dmul_rn(a, b)` for the leading
      product fixes this — DD precision is recovered.
   2. The Horner-series reciprocal constants `1/k` must be DD-precision
      (built via `dd_div(one, k)`); writing `dd_make(1.0 / k, 0.0)`
      carries only fp64 precision in the constant and collapses the
      result series.

   Expected precision: ~30+ decimal digits matching CPU libquadmath
   (the fp128 native CPU path) for every value in the tested set. CPU
   libquadmath has slightly more precision (~34 digits, native fp128)
   so the last 1-2 digits may differ.
*/

function match_digits($a, $b) {
    $n = 0;
    $minlen = min(strlen($a), strlen($b));
    for ($i = 0; $i < $minlen; $i++) {
        if ($a[$i] === $b[$i]) $n++; else break;
    }
    return $n;
}

function check_op($op, $vals, $min_digits) {
    foreach ($vals as $v) {
        $cpu = new NDArray([$v], 'float128');
        $rc = (string)NumPower::$op($cpu)->toArray()[0];
        $rg = (string)NumPower::$op($cpu->gpu())->cpu()->toArray()[0];
        /* When CPU returns the short canonical integer form ("1", "-3"),
           the GPU may produce the same value with a sub-DD-ULP residual
           ("-2.999999999999999999999999999999926"). Compare numerically
           via fp64 relative error in that case; otherwise count matching
           characters of the string forms. */
        $is_short = (strlen($rc) <= 5);
        if ($is_short) {
            $cpv = (float)$rc;
            $gpv = (float)$rg;
            $tol = abs($cpv) > 0 ? abs($cpv) * 1e-30 : 1e-30;
            if (abs($cpv - $gpv) <= $tol || $rc === $rg) {
                echo "OK $op($v) ≈ exact: $rc\n";
            } else {
                echo "FAIL $op($v): got $rg, want $rc\n";
            }
        } else {
            $m = match_digits($rc, $rg);
            if ($m >= $min_digits) {
                echo "OK $op($v) DD precision: $m digits match\n";
            } else {
                echo "FAIL $op($v): only $m digits match\n  CPU=$rc\n  GPU=$rg\n";
            }
        }
    }
}

/* Threshold = 28 decimal digits — DD ULP is at ~10⁻³², so 28 digits
   gives 4 digits of headroom for accumulated arithmetic error across
   the ~30 DD ops a transcendental performs. */
$T = 28;

/* ── exp family ────────────────────────────────────────────────────── */
check_op('exp',   ['0.5', '1.0', '2.0', '10.0', '50.0', '100.0', '500.0'], $T);
check_op('exp2',  ['0.5', '2.0', '5.0', '100.0', '0.1', '-1.0'], $T);
check_op('expm1', ['0.5', '1.0', '2.0', '10.0', '100.0', '0.001'], $T);

/* ── log family ────────────────────────────────────────────────────── */
check_op('log',   ['2.0', '0.5', '10.0', '100.0', '0.1', '1000.0', '0.001'], $T);
check_op('log1p', ['0.5', '1.0', '2.0', '10.0', '0.001', '0.0001'], $T);
check_op('log2',  ['10.0', '100.0', '0.5', '0.1', '0.001'], $T);
check_op('log10', ['2.0', '0.5', '0.1', '0.001'], $T);

/* ── Exact-result preservation ─────────────────────────────────────── */
echo "\n=== Exact integer results ===\n";
foreach ([
    ['log',   '1.0',     '0'],
    ['log2',  '1.0',     '0'],
    ['log10', '1.0',     '0'],
    ['exp',   '0.0',     '1'],
    ['exp2',  '0.0',     '1'],
    ['exp2',  '1.0',     '2'],
    ['exp2',  '10.0',    '1024'],
    ['log2',  '1024.0',  '10'],
    ['log2',  '0.5',     '-1'],
    ['log',   '0.0',     '-inf'],
    ['exp',   '-inf',    '0'],
] as [$op, $in, $want]) {
    $cpu = new NDArray([$in], 'float128');
    $rg = (string)NumPower::$op($cpu->gpu())->cpu()->toArray()[0];
    if ($rg === $want) echo "OK $op($in) = $want (exact)\n";
    else               echo "FAIL $op($in): got $rg, want $want\n";
}

/* ── Special-value propagation on GPU fp128 ────────────────────────── */
echo "\n=== Special values ===\n";
foreach ([
    ['log',  '-1.0',  'nan'],
    ['sqrt', '-1.0',  'nan'],
    ['exp',  'inf',   'inf'],
    ['log',  'inf',   'inf'],
] as [$op, $in, $want]) {
    $cpu = new NDArray([$in], 'float128');
    $rg = (string)NumPower::$op($cpu->gpu())->cpu()->toArray()[0];
    if ($rg === $want) echo "OK $op($in) = $want\n";
    else               echo "FAIL $op($in): got $rg, want $want\n";
}

echo "DONE\n";
?>
--EXPECTF--
OK exp(0.5) DD precision: %d digits match
OK exp(1.0) DD precision: %d digits match
OK exp(2.0) DD precision: %d digits match
OK exp(10.0) DD precision: %d digits match
OK exp(50.0) DD precision: %d digits match
OK exp(100.0) DD precision: %d digits match
OK exp(500.0) DD precision: %d digits match
OK exp2(0.5) DD precision: %d digits match
OK exp2(2.0) ≈ exact: 4
OK exp2(5.0) ≈ exact: 32
OK exp2(100.0) DD precision: %d digits match
OK exp2(0.1) DD precision: %d digits match
OK exp2(-1.0) ≈ exact: 0.5
OK expm1(0.5) DD precision: %d digits match
OK expm1(1.0) DD precision: %d digits match
OK expm1(2.0) DD precision: %d digits match
OK expm1(10.0) DD precision: %d digits match
OK expm1(100.0) DD precision: %d digits match
OK expm1(0.001) DD precision: %d digits match
OK log(2.0) DD precision: %d digits match
OK log(0.5) DD precision: %d digits match
OK log(10.0) DD precision: %d digits match
OK log(100.0) DD precision: %d digits match
OK log(0.1) DD precision: %d digits match
OK log(1000.0) DD precision: %d digits match
OK log(0.001) DD precision: %d digits match
OK log1p(0.5) DD precision: %d digits match
OK log1p(1.0) DD precision: %d digits match
OK log1p(2.0) DD precision: %d digits match
OK log1p(10.0) DD precision: %d digits match
OK log1p(0.001) DD precision: %d digits match
OK log1p(0.0001) DD precision: %d digits match
OK log2(10.0) DD precision: %d digits match
OK log2(100.0) DD precision: %d digits match
OK log2(0.5) ≈ exact: -1
OK log2(0.1) DD precision: %d digits match
OK log2(0.001) DD precision: %d digits match
OK log10(2.0) DD precision: %d digits match
OK log10(0.5) DD precision: %d digits match
OK log10(0.1) ≈ exact: -1
OK log10(0.001) ≈ exact: -3

=== Exact integer results ===
OK log(1.0) = 0 (exact)
OK log2(1.0) = 0 (exact)
OK log10(1.0) = 0 (exact)
OK exp(0.0) = 1 (exact)
OK exp2(0.0) = 1 (exact)
OK exp2(1.0) = 2 (exact)
OK exp2(10.0) = 1024 (exact)
OK log2(1024.0) = 10 (exact)
OK log2(0.5) = -1 (exact)
OK log(0.0) = -inf (exact)
OK exp(-inf) = 0 (exact)

=== Special values ===
OK log(-1.0) = nan
OK sqrt(-1.0) = nan
OK exp(inf) = inf
OK log(inf) = inf
DONE
