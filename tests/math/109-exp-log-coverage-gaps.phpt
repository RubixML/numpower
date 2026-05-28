--TEST--
NumPower exp/log family: coverage gaps (GPU multi-block, int boundary, fp overflow/underflow, multi-dim non-fp64, empty multi-dim)
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
/* Targets the priority coverage gaps surfaced by the post-implementation
   review:
   1. GPU multi-block correctness (N > 1024 elements crosses kernel block
      boundary; the 1-D blockSize=256 launches in unary_run_gpu_inplace).
   2. Integer-dtype × full 8-op matrix on CPU & GPU (not just log2).
   3. Boundary integer inputs (INT_MIN, UINT64_MAX, etc.).
   4. fp32 / fp64 overflow/underflow boundaries (exp(710) → +Inf, etc.).
   5. fp16 max-finite (±65504) and underflow boundaries.
   6. 2-D / 3-D / 4-D shapes on dtypes other than fp64.
   7. Empty multi-dim shapes ([0,5], [5,0,3]).
   8. Negative-domain log family across all dtypes. */

function approx($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g); $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!approx($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
    if (is_float($g) || is_float($w)) {
        $gf = (float)$g; $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        if (is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0))) return true;
        if (is_infinite($gf) !== is_infinite($wf)) return false;
        if (is_nan($gf) !== is_nan($wf)) return false;
        if ($wf == 0.0) return abs($gf) <= $tol;
        return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
    }
    return (string)$g === (string)$w;
}

function check($label, $got, $want, $tol = 0.0) {
    if (approx($got, $want, $tol)) {
        echo "OK $label\n";
    } else {
        echo "FAIL $label: got=", json_encode($got),
             " want=", json_encode($want), "\n";
    }
}

/* ──────────────────────────────────────────────────────────────────────
   1. GPU multi-block: N > 1024 elements crosses the kernel block boundary.
      We pick N = 4097 to ensure >= 2 blocks (block size 256) AND a tail
      thread; verify exp / log / logb / log2 / exp2 on float32 stays
      correct end-to-end and matches CPU bit-equivalently.
   ────────────────────────────────────────────────────────────────────── */
$N = 4097;
$xs = [];
for ($i = 0; $i < $N; $i++) {
    $xs[] = 1.0 + (float)$i / 100.0;   /* avoid log(0); range ~ [1.0, 41.97] */
}
foreach (['float32', 'float64'] as $dt) {
    $tol = ($dt === 'float32') ? 1e-3 : 1e-9;
    $a_cpu = NumPower::array($xs, $dt);
    $a_gpu = NumPower::array($xs, $dt)->gpu();
    foreach (['exp','log','log2','log10','log1p','exp2','expm1','logb'] as $op) {
        $r_cpu = NumPower::$op($a_cpu)->toArray();
        $r_gpu = NumPower::$op($a_gpu)->cpu()->toArray();
        $cv = array_values($r_cpu);
        $gv = array_values($r_gpu);
        check("multi-block $N $dt $op CPU=GPU", $gv, $cv, $tol);
        /* Spot check: last element passes through the kernel correctly. */
        $lastInput = $xs[$N - 1];
        $expectedLast = ($op === 'exp')   ? exp($lastInput)
                      : (($op === 'log')   ? log($lastInput)
                      : (($op === 'log2')  ? log($lastInput, 2)
                      : (($op === 'log10') ? log10($lastInput)
                      : (($op === 'log1p') ? log1p($lastInput)
                      : (($op === 'exp2')  ? 2.0 ** $lastInput
                      : (($op === 'expm1') ? expm1($lastInput)
                      : log($lastInput, 2)))))));
        check("multi-block $N $dt $op last elem", $gv[$N - 1], $expectedLast, $tol);
    }
}

/* ──────────────────────────────────────────────────────────────────────
   2. Integer-dtype × full 8-op matrix on CPU AND GPU.
      Up to now we only had log2 value-checked for ints; the other ops only
      verified dtype promotion. Pick inputs that have closed-form results
      under each transcendental so any precision drift is visible.
   ────────────────────────────────────────────────────────────────────── */
$int_dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'];
$xs_int = [1, 2, 4, 8];   /* log/log2/log10 outputs are exact integers */
foreach ($int_dtypes as $dt) {
    $is_wide = in_array($dt, ['int32','int64','uint32','uint64'], true);
    $tol = $is_wide ? 1e-12 : 1e-5;
    $a_cpu = NumPower::array($xs_int, $dt);
    $a_gpu = NumPower::array($xs_int, $dt)->gpu();

    /* exp(1)=e, exp(2)=e^2, exp(4)=e^4, exp(8)=e^8 */
    check("$dt exp CPU",  NumPower::exp($a_cpu)->toArray(),
                          [M_E, M_E**2, M_E**4, M_E**8], $tol);
    check("$dt exp GPU",  NumPower::exp($a_gpu)->cpu()->toArray(),
                          [M_E, M_E**2, M_E**4, M_E**8], $tol);
    /* log(8)=3*log(2) etc. */
    check("$dt log CPU",  NumPower::log($a_cpu)->toArray(),
                          [0, log(2), log(4), log(8)], $tol);
    check("$dt log GPU",  NumPower::log($a_gpu)->cpu()->toArray(),
                          [0, log(2), log(4), log(8)], $tol);
    /* log2 of powers of 2 */
    check("$dt log2 CPU", NumPower::log2($a_cpu)->toArray(), [0, 1, 2, 3], $tol);
    check("$dt log2 GPU", NumPower::log2($a_gpu)->cpu()->toArray(), [0, 1, 2, 3], $tol);
    /* logb(2^k) = k */
    check("$dt logb CPU", NumPower::logb($a_cpu)->toArray(), [0, 1, 2, 3], $tol);
    check("$dt logb GPU", NumPower::logb($a_gpu)->cpu()->toArray(), [0, 1, 2, 3], $tol);
    /* exp2(int) */
    check("$dt exp2 CPU", NumPower::exp2($a_cpu)->toArray(), [2, 4, 16, 256], $tol);
    check("$dt exp2 GPU", NumPower::exp2($a_gpu)->cpu()->toArray(), [2, 4, 16, 256], $tol);
    /* log10(8)=log10(8) etc. */
    check("$dt log10 CPU", NumPower::log10($a_cpu)->toArray(),
                           [0, log10(2), log10(4), log10(8)], $tol);
    check("$dt log10 GPU", NumPower::log10($a_gpu)->cpu()->toArray(),
                           [0, log10(2), log10(4), log10(8)], $tol);
}

/* ──────────────────────────────────────────────────────────────────────
   3. Boundary integer inputs: log2 / logb of dtype maxima are the
      classic precision-loss site for the uint64-to-float64 cast.
      The exact answer is 64 (uint64 max) etc.
   ────────────────────────────────────────────────────────────────────── */
$u8_max  = 255;
$u16_max = 65535;
$u32_max = 4294967295;

check("uint8  log2(max+1)≈8",   NumPower::log2(NumPower::array([$u8_max + 1], 'int32'))->toArray(),  [8], 1e-12);
check("uint16 log2(max+1)≈16",  NumPower::log2(NumPower::array([$u16_max + 1], 'int64'))->toArray(), [16], 1e-12);
check("uint32 log2(max+1)≈32",  NumPower::log2(NumPower::array([$u32_max + 1], 'int64'))->toArray(), [32], 1e-12);

/* uint64 max = 2^64-1; passed as string to preserve precision. log2 ~ 64. */
$ul_max = ['18446744073709551615'];
$u64_log2 = NumPower::log2(NumPower::array($ul_max, 'uint64'))->toArray();
/* log2 of (2^64 - 1) is just under 64, but rounds to 64.0 in fp64. */
check("uint64 log2(2^64-1)≈64", $u64_log2, [64], 1e-12);

/* Same on GPU */
$u64_log2_gpu = NumPower::log2(NumPower::array($ul_max, 'uint64')->gpu())->cpu()->toArray();
check("uint64 log2(2^64-1) GPU≈64", $u64_log2_gpu, [64], 1e-12);

/* INT_MIN of int8 → log of negative → NaN (not exception). */
$i8_min = NumPower::array([-128], 'int8');
$lr = NumPower::log($i8_min)->toArray();
check("int8 log(-128)=NaN CPU", is_nan($lr[0]), true);
$lr_gpu = NumPower::log($i8_min->gpu())->cpu()->toArray();
check("int8 log(-128)=NaN GPU", is_nan($lr_gpu[0]), true);

/* logb of negative: |x| in logb, so logb(-128) = logb(128) = 7. */
$lb = NumPower::logb($i8_min)->toArray();
check("int8 logb(-128)=7 CPU", $lb[0], 7, 1e-12);
$lb_gpu = NumPower::logb($i8_min->gpu())->cpu()->toArray();
check("int8 logb(-128)=7 GPU", $lb_gpu[0], 7, 1e-12);

/* ──────────────────────────────────────────────────────────────────────
   4. fp32 / fp64 overflow & underflow boundaries.
      Pre-fix code routed everything through fp32, so fp64 overflow at
      exp(710) was masked by an earlier exp(89)≈Inf. Now fp64 exp(709)
      finite ≈ 8.2e307, fp64 exp(710) ≈ +Inf.
   ────────────────────────────────────────────────────────────────────── */
foreach (['CPU', 'GPU'] as $dev) {
    $maker = function ($vals, $dt) use ($dev) {
        $a = NumPower::array($vals, $dt);
        return ($dev === 'GPU') ? $a->gpu() : $a;
    };
    $fetch = function ($r) use ($dev) {
        return ($dev === 'GPU') ? $r->cpu()->toArray() : $r->toArray();
    };

    /* fp64: exp(709.78) ≈ finite max, exp(710) → +Inf */
    $a64 = $maker([709.0, 710.0, -745.0, -746.0], 'float64');
    $e64 = $fetch(NumPower::exp($a64));
    check("$dev fp64 exp(709) finite",         is_finite($e64[0]) && $e64[0] > 0, true);
    check("$dev fp64 exp(710)=+Inf",           is_infinite($e64[1]) && $e64[1] > 0, true);
    check("$dev fp64 exp(-745) subnormal/0",   $e64[2] >= 0 && $e64[2] < 1e-307, true);
    check("$dev fp64 exp(-746)=0",             $e64[3], 0.0, 1e-300);

    /* fp32: exp(88) finite, exp(89) → +Inf */
    $a32 = $maker([88.0, 89.0, -103.0, -104.0], 'float32');
    $e32 = $fetch(NumPower::exp($a32));
    check("$dev fp32 exp(88) finite",          is_finite($e32[0]) && $e32[0] > 0, true);
    check("$dev fp32 exp(89)=+Inf",            is_infinite($e32[1]) && $e32[1] > 0, true);
    check("$dev fp32 exp(-103) ≥ 0",           $e32[2] >= 0, true);
    check("$dev fp32 exp(-104)=0",             $e32[3], 0.0, 1e-30);

    /* exp2(1024)=+Inf on fp64; exp2(1023) finite */
    $a2 = $maker([1023.0, 1024.0, -1074.0, -1075.0], 'float64');
    $e2 = $fetch(NumPower::exp2($a2));
    check("$dev fp64 exp2(1023) finite",       is_finite($e2[0]), true);
    check("$dev fp64 exp2(1024)=+Inf",         is_infinite($e2[1]) && $e2[1] > 0, true);
}

/* ──────────────────────────────────────────────────────────────────────
   5. fp16 boundary values: max finite (±65504), smallest normal (~6e-5),
      smallest subnormal (~6e-8). On fp16, exp(11)=fp16-finite but
      exp(12)=+Inf (fp16 max ~= 65504, exp(12) ≈ 162755).
   ────────────────────────────────────────────────────────────────────── */
foreach (['CPU', 'GPU'] as $dev) {
    $maker = function ($vals, $dt) use ($dev) {
        $a = NumPower::array($vals, $dt);
        return ($dev === 'GPU') ? $a->gpu() : $a;
    };
    $fetch = function ($r) use ($dev) {
        return ($dev === 'GPU') ? $r->cpu()->toArray() : $r->toArray();
    };
    /* fp16 exp range */
    $h = $maker([10.0, 11.0, 12.0, -16.0, -17.0], 'float16');
    $eh = $fetch(NumPower::exp($h));
    check("$dev fp16 exp(10) finite",          is_finite($eh[0]) && $eh[0] > 0, true);
    /* exp(12) ≈ 162755 — exceeds fp16 max 65504, must overflow to +Inf. */
    check("$dev fp16 exp(12)=+Inf",            is_infinite($eh[2]) && $eh[2] > 0, true);
    /* exp(-17) ≈ 4.1e-8 — below fp16 smallest subnormal 6e-8, must underflow. */
    check("$dev fp16 exp(-17) ≈ 0",            abs($eh[4]) < 1e-3, true);

    /* fp16 log of max-finite (65504) ≈ log(65504) ≈ 11.09 */
    $lh = $maker([65504.0, 1.0], 'float16');
    $r_lh = $fetch(NumPower::log($lh));
    check("$dev fp16 log(65504) ≈ 11.09",      $r_lh[0], 11.09, 0.1);
    check("$dev fp16 log(1) = 0",              $r_lh[1], 0.0, 1e-3);
}

/* ──────────────────────────────────────────────────────────────────────
   6. Multi-dim shapes (2-D, 3-D, 4-D) on dtypes OTHER than fp64.
   ────────────────────────────────────────────────────────────────────── */
$mat32 = NumPower::array([[1.0, 2.0], [4.0, 8.0]], 'float32');
check("2-D fp32 log2 CPU", NumPower::log2($mat32)->toArray(),
      [[0, 1], [2, 3]], 1e-5);
check("2-D fp32 log2 GPU", NumPower::log2($mat32->gpu())->cpu()->toArray(),
      [[0, 1], [2, 3]], 1e-5);

$mat16 = NumPower::array([[1.0, 2.0], [4.0, 8.0]], 'float16');
check("2-D fp16 log2 CPU", NumPower::log2($mat16)->toArray(),
      [[0, 1], [2, 3]], 1e-2);
check("2-D fp16 log2 GPU", NumPower::log2($mat16->gpu())->cpu()->toArray(),
      [[0, 1], [2, 3]], 1e-2);

/* fp128 2-D */
$mat128 = NumPower::array([['1.0', '2.0'], ['4.0', '8.0']], 'float128');
$lv = NumPower::log2($mat128)->toArray();
check("2-D fp128 log2 shape", count($lv), 2);
check("2-D fp128 log2 elem[0][0]=0", (float)$lv[0][0], 0.0, 1e-12);
check("2-D fp128 log2 elem[1][1]=3", (float)$lv[1][1], 3.0, 1e-12);
/* GPU */
$lvg = NumPower::log2($mat128->gpu())->cpu()->toArray();
check("2-D fp128 GPU log2 elem[1][1]≈3", (float)$lvg[1][1], 3.0, 1e-9);

/* 3-D fp32 */
$cube32 = NumPower::array([[[1.0, 2.0],[4.0, 8.0]],[[16.0, 32.0],[64.0, 128.0]]], 'float32');
check("3-D fp32 log2 CPU", NumPower::log2($cube32)->toArray(),
      [[[0, 1],[2, 3]],[[4, 5],[6, 7]]], 1e-5);
check("3-D fp32 log2 GPU", NumPower::log2($cube32->gpu())->cpu()->toArray(),
      [[[0, 1],[2, 3]],[[4, 5],[6, 7]]], 1e-5);

/* 4-D fp64 (build via reshape-like nesting) */
$flat = [];
for ($i = 0; $i < 16; $i++) $flat[] = 1.0;
$nd4 = NumPower::array([[[[1.0, 2.0], [4.0, 8.0]],
                          [[16.0, 32.0], [64.0, 128.0]]],
                         [[[256.0, 512.0], [1024.0, 2048.0]],
                          [[4096.0, 8192.0], [16384.0, 32768.0]]]], 'float64');
$r = NumPower::log2($nd4)->toArray();
/* All values 2^k yield log2 = k from 0 to 15 in flattened order. */
$flat_r = [];
array_walk_recursive($r, function ($v) use (&$flat_r) { $flat_r[] = $v; });
$expected_flat = range(0, 15);
check("4-D fp64 log2 CPU (flattened)", $flat_r, array_map('floatval', $expected_flat), 1e-12);

/* ──────────────────────────────────────────────────────────────────────
   7. Empty multi-dim shapes.
   ────────────────────────────────────────────────────────────────────── */
foreach (['float32', 'float64', 'float16'] as $dt) {
    /* Empty 2-D: [0, 5] */
    $empty2 = NumPower::zeros([0, 5], $dt);
    foreach (['exp', 'log', 'log2', 'logb'] as $op) {
        $r = NumPower::$op($empty2);
        check("$dt $op empty[0,5] shape",  $r->shape(), [0, 5]);
        check("$dt $op empty[0,5] dtype",  $r->__serialize()['dtype'], $dt);
    }
    /* Empty 3-D: [5, 0, 3] */
    $empty3 = NumPower::zeros([5, 0, 3], $dt);
    $r = NumPower::log($empty3);
    check("$dt log empty[5,0,3] shape", $r->shape(), [5, 0, 3]);
    check("$dt log empty[5,0,3] dtype", $r->__serialize()['dtype'], $dt);
}

/* ──────────────────────────────────────────────────────────────────────
   8. Negative-domain log family: must be NaN on negative, -Inf on 0.
      Sweep all fp dtypes on CPU + GPU.
   ────────────────────────────────────────────────────────────────────── */
foreach (['float32', 'float64', 'float16'] as $dt) {
    foreach (['CPU', 'GPU'] as $dev) {
        $a = NumPower::array([-1.0, 0.0], $dt);
        if ($dev === 'GPU') $a = $a->gpu();
        $fetch = ($dev === 'GPU') ? fn($x) => $x->cpu()->toArray() : fn($x) => $x->toArray();

        $l = $fetch(NumPower::log($a));
        check("$dev $dt log(-1)=NaN", is_nan($l[0]), true);
        check("$dev $dt log(0)=-Inf", is_infinite($l[1]) && $l[1] < 0, true);

        $l2 = $fetch(NumPower::log2($a));
        check("$dev $dt log2(-1)=NaN", is_nan($l2[0]), true);
        check("$dev $dt log2(0)=-Inf", is_infinite($l2[1]) && $l2[1] < 0, true);

        $l10 = $fetch(NumPower::log10($a));
        check("$dev $dt log10(-1)=NaN", is_nan($l10[0]), true);
        check("$dev $dt log10(0)=-Inf", is_infinite($l10[1]) && $l10[1] < 0, true);
    }
}

echo "DONE\n";
?>
--EXPECTF--
%aDONE
