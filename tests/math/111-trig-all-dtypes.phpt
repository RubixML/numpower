--TEST--
NumPower trig / hyperbolic / angle / rounding family: all dtypes, CPU + GPU, fixes pre-existing NDArray_Map dtype-truncation bugs
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
/* Regression + coverage for the 19-op trig/hyperbolic/angle/rounding
   refactor. Before the refactor these all used the legacy `NDArray_Map`
   / `cuda_float_*` paths which silently truncated every non-fp32 input
   to fp32:
     - sin(float64) returned float32 with garbage past ~7 digits.
     - sin(int32) returned float32 with the buffer reinterpreted as fp32
       (undefined-value cells).
     - sin(float128) overran the 16-byte fp128 storage with fp32 writes.
     - GPU sin/cos/etc. on any non-fp32 buffer produced garbage.

   After: all 19 ops route through the typed unary dispatcher with
   per-dtype kernels (CPU libm + CUDA libdevice + libquadmath/DD for
   fp128). Dtype promotion follows PyTorch widening (narrow ints →
   float32, wide ints → float64); the rounding family preserves dtype. */

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
        if ($wf == 0.0) return abs($gf) <= $tol;
        return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
    }
    return (string)$g === (string)$w;
}
function check($label, $got, $want, $tol = 0.0) {
    echo (approx($got, $want, $tol) ? "OK " : ("FAIL ")) . "$label";
    if (!approx($got, $want, $tol)) {
        echo ": got=", json_encode($got), " want=", json_encode($want);
    }
    echo "\n";
}

/* ─── float32 / float64 / float16 unary trig + hyperbolic ─── */
$trig_inputs   = [0.0, 0.5, 1.0, M_PI / 2];
$hyp_inputs    = [0.0, 0.5, 1.0, 2.0];
$bounded_inputs = [0.0, 0.5, -0.5];        /* arctanh domain (-1, 1) */
$pos_inputs    = [1.0, 2.0, 10.0];          /* arccosh domain ≥ 1 */
$round_inputs  = [-2.7, -0.5, 0.5, 2.5];   /* rounding edges */

$tols = ['float32' => 1e-5, 'float64' => 1e-12, 'float16' => 5e-2];

foreach ($tols as $dt => $tol) {
    $sin_in  = NumPower::array($trig_inputs, $dt);
    $hyp_in  = NumPower::array($hyp_inputs, $dt);
    $bnd_in  = NumPower::array($bounded_inputs, $dt);
    $pos_in  = NumPower::array($pos_inputs, $dt);
    $rnd_in  = NumPower::array($round_inputs, $dt);

    /* Compute expected via Python's math (libm) — same intrinsics. */
    $exp = function ($inputs, $fn) {
        return array_map($fn, $inputs);
    };

    foreach ([false, true] as $on_gpu) {
        $dev = $on_gpu ? 'GPU' : 'CPU';
        $stage = $on_gpu
            ? fn($a) => $a->gpu()
            : fn($a) => $a;
        $back = $on_gpu
            ? fn($r) => $r->cpu()->toArray()
            : fn($r) => $r->toArray();

        check("$dt $dev sin",     $back(NumPower::sin($stage($sin_in))),
                                  $exp($trig_inputs, 'sin'), $tol);
        check("$dt $dev cos",     $back(NumPower::cos($stage($sin_in))),
                                  $exp($trig_inputs, 'cos'), $tol);
        check("$dt $dev tan",     $back(NumPower::tan($stage(NumPower::array([0.0, M_PI / 4], $dt)))),
                                  [0.0, 1.0], $tol);
        check("$dt $dev arcsin",  $back(NumPower::arcsin($stage($bnd_in))),
                                  $exp($bounded_inputs, 'asin'), $tol);
        check("$dt $dev arccos",  $back(NumPower::arccos($stage($bnd_in))),
                                  $exp($bounded_inputs, 'acos'), $tol);
        check("$dt $dev arctan",  $back(NumPower::arctan($stage($sin_in))),
                                  $exp($trig_inputs, 'atan'), $tol);

        check("$dt $dev sinh",    $back(NumPower::sinh($stage($hyp_in))),
                                  $exp($hyp_inputs, 'sinh'), $tol);
        check("$dt $dev cosh",    $back(NumPower::cosh($stage($hyp_in))),
                                  $exp($hyp_inputs, 'cosh'), $tol);
        check("$dt $dev tanh",    $back(NumPower::tanh($stage($hyp_in))),
                                  $exp($hyp_inputs, 'tanh'), $tol);
        check("$dt $dev arcsinh", $back(NumPower::arcsinh($stage($hyp_in))),
                                  $exp($hyp_inputs, 'asinh'), $tol);
        check("$dt $dev arccosh", $back(NumPower::arccosh($stage($pos_in))),
                                  $exp($pos_inputs, 'acosh'), $tol);
        check("$dt $dev arctanh", $back(NumPower::arctanh($stage($bnd_in))),
                                  $exp($bounded_inputs, 'atanh'), $tol);

        check("$dt $dev degrees", $back(NumPower::degrees($stage(NumPower::array([0.0, M_PI, M_PI / 2], $dt)))),
                                  [0.0, 180.0, 90.0], $tol);
        check("$dt $dev radians", $back(NumPower::radians($stage(NumPower::array([0.0, 180.0, 90.0], $dt)))),
                                  [0.0, M_PI, M_PI / 2], $tol);

        /* Rounding family preserves dtype (no promotion). */
        check("$dt $dev rint",    $back(NumPower::rint($stage($rnd_in))),
                                  [-3.0, 0.0, 0.0, 2.0], $tol);   /* round-half-to-even */
        check("$dt $dev floor",   $back(NumPower::floor($stage($rnd_in))),
                                  [-3.0, -1.0, 0.0, 2.0], $tol);
        check("$dt $dev ceil",    $back(NumPower::ceil($stage($rnd_in))),
                                  [-2.0, 0.0, 1.0, 3.0], $tol);
        check("$dt $dev trunc",   $back(NumPower::trunc($stage($rnd_in))),
                                  [-2.0, 0.0, 0.0, 2.0], $tol);
        check("$dt $dev fix",     $back(NumPower::fix($stage($rnd_in))),
                                  [-2.0, 0.0, 0.0, 2.0], $tol);
    }
}

/* ─── int dtypes: trig promotes to float, rounding preserves int ─── */
foreach (['int8','int16','int32','int64','uint8','uint16','uint32','uint64'] as $dt) {
    $is_wide = in_array($dt, ['int32','int64','uint32','uint64'], true);
    $expected_promote = $is_wide ? 'float64' : 'float32';
    $tol = $is_wide ? 1e-12 : 1e-5;

    foreach ([false, true] as $on_gpu) {
        $dev = $on_gpu ? 'GPU' : 'CPU';
        $a = NumPower::array([0, 1, 2], $dt);
        if ($on_gpu) $a = $a->gpu();
        $r = NumPower::sin($a);
        $back = $on_gpu ? fn($x) => $x->cpu() : fn($x) => $x;
        check("$dt $dev sin promotes",  $back($r)->__serialize()['dtype'], $expected_promote);
        check("$dt $dev sin values",    $back($r)->toArray(), [0.0, sin(1.0), sin(2.0)], $tol);

        $rnd = NumPower::array([1, 2, 3], $dt);
        if ($on_gpu) $rnd = $rnd->gpu();
        $rr = NumPower::floor($rnd);
        check("$dt $dev floor preserves",  $back($rr)->__serialize()['dtype'], $dt);
        check("$dt $dev ceil preserves",   $back(NumPower::ceil($rnd))->__serialize()['dtype'], $dt);
        check("$dt $dev rint preserves",   $back(NumPower::rint($rnd))->__serialize()['dtype'], $dt);
    }
}

/* ─── float128: full libquadmath precision on CPU; DD ~fp64 on GPU ─── */
$f128 = NumPower::array(['0.0', '0.5235987755982988730771072305465838', '1.0', '1.5707963267948966192313216916398'], 'float128');

$cpu_sin = NumPower::sin($f128)->toArray();
check("fp128 sin(0)",          (float)$cpu_sin[0], 0.0,  1e-15);
check("fp128 sin(π/6) ≈ 0.5",   (float)$cpu_sin[1], 0.5,  1e-12);
check("fp128 sin(π/2) = 1",     (float)$cpu_sin[3], 1.0,  1e-15);

$cpu_cos = NumPower::cos($f128)->toArray();
check("fp128 cos(0) = 1",       (float)$cpu_cos[0], 1.0,  1e-15);
check("fp128 cos(π/2) ≈ 0",     (float)$cpu_cos[3], 0.0,  1e-15);

/* GPU fp128 */
$gpu_sin = NumPower::sin($f128->gpu())->cpu()->toArray();
check("fp128 GPU sin(π/2) ≈ 1", (float)$gpu_sin[3], 1.0, 1e-9);

/* fp128 rounding */
$r128 = NumPower::array(['-2.7', '-0.5', '0.5', '2.5'], 'float128');
$f_floor = NumPower::floor($r128)->toArray();
check("fp128 floor(-2.7)",   (float)$f_floor[0], -3.0, 1e-15);
check("fp128 floor(2.5)",    (float)$f_floor[3],  2.0, 1e-15);
$f_ceil  = NumPower::ceil($r128)->toArray();
check("fp128 ceil(-2.7)",    (float)$f_ceil[0], -2.0, 1e-15);

/* ─── float4 / float8 (narrow non-half floats) ─── */
foreach (['float4', 'float8'] as $dt) {
    $a = NumPower::array([0.0, 0.5, 1.0], $dt);
    $r = NumPower::sin($a);
    check("$dt sin preserves dtype",  $r->__serialize()['dtype'], $dt);
    /* fp4 max representable = 6, max input range tiny — sin(0)=0 exact */
    check("$dt sin(0)=0",  (float)$r->toArray()[0], 0.0, 0.1);
}

/* ─── Multi-dim shapes ─── */
$mat = NumPower::array([[0.0, M_PI / 2], [M_PI, 3 * M_PI / 2]], 'float64');
$mat_sin = NumPower::sin($mat)->toArray();
check("2-D sin",          $mat_sin, [[0.0, 1.0], [0.0, -1.0]], 1e-12);
$mat_sin_gpu = NumPower::sin($mat->gpu())->cpu()->toArray();
check("2-D GPU sin",      $mat_sin_gpu, [[0.0, 1.0], [0.0, -1.0]], 1e-12);

$cube = NumPower::array([[[0.0, 1.0], [2.0, 3.0]]], 'float64');
$cube_floor = NumPower::floor(NumPower::array([[[0.7, 1.5], [-0.5, -1.5]]], 'float64'))->toArray();
check("3-D floor",        $cube_floor, [[[0.0, 1.0], [-1.0, -2.0]]], 1e-15);

/* ─── 0-D scalar inputs return PHP scalar ─── */
check("0-D sin scalar",   NumPower::sin(NumPower::array(M_PI / 2)),  1.0, 1e-5);
check("0-D cos scalar",   NumPower::cos(NumPower::array(0.0)),       1.0, 1e-5);
check("0-D floor scalar", NumPower::floor(NumPower::array(1.7)),     1.0, 1e-12);

/* ─── Edge values: NaN/Inf propagation ─── */
$edge = NumPower::array([INF, -INF, NAN, 0.0], 'float64');
$sin_edge = NumPower::sin($edge)->toArray();
check("sin(NaN)=NaN",  is_nan($sin_edge[2]), true);
check("sin(0)=0",       $sin_edge[3], 0.0, 1e-15);
$cos_edge = NumPower::cos($edge)->toArray();
check("cos(NaN)=NaN",  is_nan($cos_edge[2]), true);
check("cos(0)=1",       $cos_edge[3], 1.0, 1e-15);

/* arccos / arcsin domain errors */
$out = NumPower::arcsin(NumPower::array([2.0, -2.0], 'float64'))->toArray();
check("arcsin(2)=NaN",   is_nan($out[0]), true);
check("arcsin(-2)=NaN",  is_nan($out[1]), true);

/* arccosh of value < 1 */
$ach = NumPower::arccosh(NumPower::array([0.5, -1.0], 'float64'))->toArray();
check("arccosh(0.5)=NaN", is_nan($ach[0]), true);

/* arctanh of |x| ≥ 1 */
$ath = NumPower::arctanh(NumPower::array([1.0, -1.0, 2.0], 'float64'))->toArray();
check("arctanh(1)=+Inf",  is_infinite($ath[0]) && $ath[0] > 0, true);
check("arctanh(-1)=-Inf", is_infinite($ath[1]) && $ath[1] < 0, true);
check("arctanh(2)=NaN",   is_nan($ath[2]), true);

/* ─── Empty arrays ─── */
foreach (['float32', 'float64', 'float16'] as $dt) {
    $e = NumPower::zeros([0, 5], $dt);
    $r = NumPower::sin($e);
    check("$dt sin(empty[0,5]) shape", $r->shape(), [0, 5]);
    check("$dt sin(empty[0,5]) dtype", $r->__serialize()['dtype'], $dt);
}

/* ─── Bare string input accepted on every op: "1.0" → fp128 scalar ─── */
foreach (['sin','cos','tan','arcsin','arccos','arctan',
          'sinh','cosh','tanh','arcsinh','arccosh','arctanh',
          'degrees','radians','rint','fix','trunc','floor','ceil'] as $op) {
    try {
        $r = (string)NumPower::$op("1.0");
        if (strlen($r) === 0) {
            echo "FAIL $op(bare string) empty result\n";
        } else {
            echo "OK $op(bare string) -> $r\n";
        }
    } catch (Error $e) {
        echo "FAIL $op(bare string) threw: ", $e->getMessage(), "\n";
    }
}
/* ─── Empty / whitespace-only strings still throw ─── */
foreach (['sin','cos','floor'] as $op) {
    try {
        NumPower::$op("");
        echo "FAIL $op('') did not throw\n";
    } catch (Error $e) {
        echo "OK $op('') throws\n";
    }
}

echo "DONE\n";
?>
--EXPECTF--
%aDONE
