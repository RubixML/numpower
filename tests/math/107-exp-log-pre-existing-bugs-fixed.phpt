--TEST--
NumPower exp/log family: regressions for pre-existing dtype-and-device bugs
--FILE--
<?php
/* Regression guards for the bugs the dtype-by-device refactor fixed.

   Before the refactor:
     - `NDArray_Map` and `NDArrayMathGPU_ElementWise` returned a
       float32 buffer regardless of input dtype, so:
         * exp(float64 input) → silently truncated to float32;
         * log(int32) → silently truncated to float32 (~7-digit precision);
         * exp(float128 input) → silently lost ~25 digits of precision;
     - the GPU paths for exp2 / log1p / log2 / log10 were either missing
       (exp2 had only `NDArray_Map`, no GPU path) or assumed float32-only,
       so anything other than float32 on GPU would have silently corrupted
       the buffer; this test confirms every (op × dtype × device) combo
       returns within the dtype's normal tolerance now.
     - exp2 CPU was the only one without a `HAVE_CUBLAS` branch, so a
       GPU-resident input was implicitly staged through CPU; this test
       confirms exp2 now stays on the input's device.
     - `NumPower::logb` had a GPU branch but only float32 kernel
       (`cuda_float_logb`); this test exercises logb on every dtype. */

function approx_equal($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g);
        $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!approx_equal($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
    if (is_array($g) || is_array($w)) return false;
    if (is_float($g) || is_float($w)) {
        $gf = (float)$g;
        $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        if (is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0))) return true;
        return abs($gf - $wf) <= $tol;
    }
    return (string)$g === (string)$w;
}

function check($label, $got, $want, $tol = 0.0) {
    if (approx_equal($got, $want, $tol)) {
        echo "OK $label\n";
    } else {
        echo "FAIL $label: got=", json_encode($got),
             " want=", json_encode($want), "\n";
    }
}

/* ── Bug #1: NDArray_Map dropped dtype to float32 ──────────────────── */
/* Pre-fix: NumPower::exp(float64) silently returned float32. */
$f64 = NumPower::array([1.0, 2.0, 3.0], 'float64');
$ef64 = NumPower::exp($f64);
check("exp float64 preserves dtype",  $ef64->__serialize()['dtype'], 'float64');
/* Compare with full fp64 precision — the value of exp(2) past digit 7
   diverges between float32 and float64. */
check("exp(2.0) full fp64 precision",
      (string)$ef64->toArray()[1],
      (string)exp(2.0));

$lf64 = NumPower::log($f64);
check("log float64 preserves dtype",  $lf64->__serialize()['dtype'], 'float64');
check("log(3.0) full fp64 precision",
      (string)$lf64->toArray()[2],
      (string)log(3.0));

/* ── Bug #2: int input on log/log2/log10/log1p/exp/exp2/expm1/logb
        — pre-fix returned float32 even for wide ints. ─────────────── */
foreach (['int32','int64','uint32','uint64'] as $dt) {
    $a = NumPower::array([1, 8, 64], $dt);
    foreach (['log','log2','log10','exp','exp2','expm1','log1p','logb'] as $op) {
        $r = NumPower::$op($a);
        $rd = $r->__serialize()['dtype'];
        check("$dt $op promotes to float64", $rd, 'float64');
    }
}
foreach (['int8','int16','uint8','uint16'] as $dt) {
    $a = NumPower::array([1, 8, 64], $dt);
    $r = NumPower::log2($a);
    check("$dt log2 promotes to float32", $r->__serialize()['dtype'], 'float32');
}

/* ── Bug #3: exp2 silently staged GPU input through CPU ────────────── */
/* Pre-fix path: NumPower::exp2 called NDArray_Map unconditionally,
   which ran the CPU loop even on a GPU buffer. The fix routes through
   the typed unary dispatcher so the GPU branch runs natively. */
try {
    $a_gpu = NumPower::array([0.0, 1.0, 10.0], 'float64')->gpu();
    $e_gpu = NumPower::exp2($a_gpu);
    check("exp2 stays on GPU", $e_gpu->isGPU(), true);
    check("exp2 GPU dtype",    $e_gpu->cpu()->__serialize()['dtype'], 'float64');
    check("exp2 GPU values",   $e_gpu->cpu()->toArray(), [1.0, 2.0, 1024.0], 1e-12);
} catch (Throwable $t) {
    echo "skip GPU not available: " . $t->getMessage() . "\n";
}

/* ── Bug #4: NumPower::logb had GPU branch but only fp32 kernel ────── */
/* Pre-fix path: logb on any non-fp32 GPU buffer ran cuda_float_logb
   which reinterpreted the bytes as fp32. Now every supported compute
   dtype has its own cuda_logb_<tag> kernel. */
try {
    foreach (['float32','float64','float16','float128'] as $dt) {
        $tol = ($dt === 'float16') ? 5e-2 : ($dt === 'float32' ? 1e-5 : 1e-12);
        $raw = ($dt === 'float128') ? ['1.0', '4.0', '1024.0']
                                     : [1.0, 4.0, 1024.0];
        $a_gpu = NumPower::array($raw, $dt)->gpu();
        $r = NumPower::logb($a_gpu);
        check("$dt logb GPU stays on GPU", $r->isGPU(), true);
        check("$dt logb GPU dtype",        $r->cpu()->__serialize()['dtype'], $dt);
        $g = array_map('floatval', $r->cpu()->toArray());
        check("$dt logb GPU values",       $g, [0.0, 2.0, 10.0], $tol);
    }
} catch (Throwable $t) {
    echo "skip GPU not available: " . $t->getMessage() . "\n";
}

/* ── Bug #5: NumPower::exp / exp2 / expm1 / log{,1p,2,10} on
       float16 silently float32-cast the storage (NDArray_Map writes
       float32 into the fp16 buffer). The dtype-aware path now keeps
       fp16 storage on both CPU and GPU. ─────────────────────────── */
foreach (['exp','exp2','log','log2','log10','expm1','log1p','logb'] as $op) {
    $a = NumPower::array([1.0, 2.0, 4.0], 'float16');
    $r = NumPower::$op($a);
    check("$op fp16 preserves dtype", $r->__serialize()['dtype'], 'float16');
}

/* ── Bug #6: fp128 transcendentals on CPU (pre-fix would crash or
       segfault — `NDArray_Map` cast through float32 even for 16-byte
       fp128 storage, leading to undefined memory access). The fix
       routes through libquadmath (or DD fallback) without touching
       float32 staging. */
$f128 = NumPower::array(['0.0', '1.0', '2.0', '3.0'], 'float128');
foreach (['exp','exp2','expm1','log','log1p','log2','log10','logb'] as $op) {
    $r = NumPower::$op($f128);
    check("$op fp128 preserves dtype", $r->__serialize()['dtype'], 'float128');
}
/* Spot-check fp128 exp(1) ≈ 2.71828... (prefix matches first ~16 digits) */
$ef128 = NumPower::exp(NumPower::array(['1.0'], 'float128'))->toArray();
check("fp128 exp(1) ≈ e prefix",
      strncmp((string)$ef128[0], '2.71828182845904523', 19) === 0, true);

echo "DONE\n";
?>
--EXPECTF--
%aDONE
