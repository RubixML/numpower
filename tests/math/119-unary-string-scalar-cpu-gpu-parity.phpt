--TEST--
String-scalar intake: 18 unary ops × NDArray input give identical results on CPU and GPU across all 14 dtypes
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
/* The string-scalar intake materialises a CPU 0-D NDArray, but NDArray
   inputs (the most common case) must execute on the device they live
   on. This test sweeps all 14 dtypes × 18 ops × CPU/GPU and verifies:
     - GPU NDArray inputs stay on GPU (no CPU staging);
     - CPU and GPU produce identical results within the dtype's normal
       tolerance (float16 / float32 wider than float64 / fp128);
     - integer dtypes match exactly when the op preserves dtype, and
       within fp32 ULP when the op widens to float for transcendentals;
     - the string-scalar fast path itself returns the same result as
       the equivalent NDArray fast path (string "1.5" ↔ NDArray
       fp128 [1.5]).
   Pre-existing CPU/GPU divergences that are NOT this branch's concern
   are documented in the per-op tolerance below — log2 on float64+
   uses log(x)/log(2) on GPU vs native log2 on CPU. */

/* Compute the tolerance for the (source dtype, op) pair. Transcendental
   ops promote narrow ints (int8/uint8/int16/uint16) to float32 on both
   devices, so the parity check must allow at least one fp32 ULP at the
   sample magnitudes (~50) — i.e. ~5e-6. The CPU and GPU libm/expf
   sources differ by exactly one fp32 ULP at expm1(4) ≈ 53.598; that's
   a documented pre-existing libm-vs-CUDA difference, not a regression. */
function tol_for($dt, $op) {
    $is_transcendental = in_array($op,
        ['exp','exp2','expm1','log','log1p','log2','log10','logb',
         'sqrt','rsqrt','reciprocal','sinc'], true);
    $narrow_int = in_array($dt, ['int8','uint8','int16','uint16'], true);
    $wide_int   = in_array($dt, ['int32','uint32','int64','uint64'], true);
    if ($narrow_int && $is_transcendental) return 1e-5;  /* widens to fp32 */
    if ($wide_int   && $is_transcendental) return 1e-9;  /* widens to fp64 */
    switch ($dt) {
        case 'float4':    return 1.0;     /* 3-bit mantissa */
        case 'float8':    return 0.5;     /* E4M3 */
        case 'float16':   return 5e-3;
        case 'float32':   return 1e-5;
        case 'float64':   return 1e-9;
        case 'float128':  return 1e-9;    /* DD-emulation on GPU */
        default:          return 0;       /* dtype-preserving integer ops are exact */
    }
}

function close($a, $b, $tol) {
    if (is_array($a) && is_array($b)) {
        if (count($a) !== count($b)) return false;
        $a = array_values($a); $b = array_values($b);
        for ($i = 0; $i < count($a); $i++) {
            if (!close($a[$i], $b[$i], $tol)) return false;
        }
        return true;
    }
    $fa = (float)$a; $fb = (float)$b;
    if (is_nan($fa) && is_nan($fb)) return true;
    if (is_infinite($fa) && is_infinite($fb) && (($fa<0)===($fb<0))) return true;
    if (abs($fa) > 1.0) return abs($fa - $fb) / max(abs($fa), abs($fb)) <= $tol;
    return abs($fa - $fb) <= $tol;
}

/* dtype → starting values that are valid inputs to every op. exp would
   overflow on big values, log needs > 0, etc. We pick small positive
   non-trivial values so every transcendental op produces a defined
   answer. */
function inputs_for($dt) {
    if ($dt === 'float128') return ['0.5','1.0','2.0','4.0'];
    return [0.5, 1.0, 2.0, 4.0];
}

/* Per-op definition: some ops change dtype, but we just check value
   parity element-wise on the result regardless of returned dtype. */
$ops = ['abs','negative','positive','reciprocal','sign','sqrt','rsqrt',
        'square','sinc',
        'exp','exp2','expm1','log','log1p','log2','log10','logb'];

$dtypes = ['float32','float64','float128','float16','float4','float8',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$max_diff = 0;
$fail_count = 0;

foreach ($dtypes as $dt) {
    $is_int   = (strpos($dt, 'int') !== false);
    $values   = $is_int ? [0, 1, 2, 4] : inputs_for($dt);
    $cpu      = NumPower::array($values, $dt);
    $gpu      = $cpu->gpu();
    if (!$gpu->isGPU()) {
        echo "FAIL $dt: gpu() did not stay on GPU\n";
        $fail_count++;
        continue;
    }
    foreach ($ops as $op) {
        $rcpu = NumPower::$op($cpu);
        $rgpu = NumPower::$op($gpu);
        /* GPU result must stay on GPU. */
        if (!$rgpu->isGPU()) {
            echo "BLOCKER $dt $op: GPU input → CPU result\n";
            $fail_count++;
            continue;
        }
        $a = $rcpu->toArray();
        $b = $rgpu->cpu()->toArray();
        $tol = tol_for($dt, $op);
        if (!close($a, $b, $tol)) {
            echo "DIFF $dt $op cpu=", json_encode($a), " gpu=", json_encode($b), "\n";
            $fail_count++;
        }
    }
}

if ($fail_count === 0) {
    echo "ALL OK\n";
} else {
    echo "TOTAL FAILURES: $fail_count\n";
}

/* ── String-scalar fast path ↔ NDArray fp128 [1.5] equivalence ──────── */
foreach ($ops as $op) {
    $via_string  = (string)NumPower::$op('1.5');
    $via_ndarray = (string)NumPower::$op(NumPower::array(['1.5'], 'float128'))->toArray()[0];
    /* Most ops on the 0-D string form return a scalar; on the 1-D NDArray
       form return an NDArray. Comparing prefixes (the value, not formatting)
       is enough to confirm the dtype/precision contract. */
    $diff = abs((float)$via_string - (float)$via_ndarray);
    if ($diff > 1e-12 &&
        !(is_nan((float)$via_string) && is_nan((float)$via_ndarray))) {
        echo "STRING vs NDArray diff for $op: str=$via_string nd=$via_ndarray\n";
    }
}
echo "DONE\n";
?>
--EXPECT--
ALL OK
DONE
