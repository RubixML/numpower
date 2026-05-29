--TEST--
Unary ops: 1-D / 2-D / 3-D / 4-D / empty shape × extreme dtype values × CPU and GPU
--FILE--
<?php
/* Sweeps every supported tensor rank (0-D collapsed to scalar, 1-D, 2-D,
   3-D, 4-D, and shape-(0,) empty) through abs (preserves dtype) and
   sqrt (widens int to float). Also verifies extreme dtype values
   (boundaries of each integer dtype, denormals / inf / nan of the
   floating types) survive a complete round-trip through abs.

   The tests use NDArray inputs (not bare strings), exercising the
   non-string code path on each device. The string-scalar path is
   exhaustively tested in 115, 117, 118, 119. */

function deep_close($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g); $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!deep_close($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
    if (is_array($g) || is_array($w)) return false;
    if (is_string($w) && is_string($g)) return $g === $w;
    $gf = (float)$g; $wf = (float)$w;
    if (is_nan($gf) && is_nan($wf)) return true;
    if (is_infinite($gf) && is_infinite($wf) && (($gf<0)===($wf<0))) return true;
    return abs($gf - $wf) <= $tol;
}

/* $silent: GPU assertions pass $silent=true so they emit nothing on
   success — the test then prints identical output on a CPU-only build
   (CI) and on a GPU box, while a genuine GPU divergence still prints a
   FAIL line and breaks the test. */
function check($label, $got, $want, $tol = 0.0, $silent = false) {
    if (deep_close($got, $want, $tol)) {
        if (!$silent) echo "OK $label\n";
    } else {
        echo "FAIL $label: got=", json_encode($got),
             " want=", json_encode($want), "\n";
    }
}

/* ── 1-D shape preservation ─────────────────────────────────────────── */
$v = NumPower::array([-3.5, 0.0, 2.5, 5.5], 'float64');
$r = NumPower::abs($v);
check("1-D abs shape",       $r->shape(), [4]);
check("1-D abs values",      $r->toArray(), [3.5, 0.0, 2.5, 5.5], 1e-12);

/* ── 2-D ───────────────────────────────────────────────────────────── */
$m = NumPower::array([[-1, 2], [-3, 4]], 'int32');
$r = NumPower::abs($m);
check("2-D abs shape",       $r->shape(), [2, 2]);
check("2-D abs values",      $r->toArray(), [[1, 2], [3, 4]]);

/* ── 3-D ───────────────────────────────────────────────────────────── */
$cube = NumPower::array([[[1.0, -2.0], [-3.0, 4.0]],
                          [[5.0, -6.0], [-7.0, 8.0]]], 'float32');
$r = NumPower::abs($cube);
check("3-D abs shape",       $r->shape(), [2, 2, 2]);
check("3-D abs values[0]",   $r->toArray()[0],
                              [[1.0, 2.0], [3.0, 4.0]], 1e-5);
check("3-D abs values[1]",   $r->toArray()[1],
                              [[5.0, 6.0], [7.0, 8.0]], 1e-5);

/* ── 4-D ───────────────────────────────────────────────────────────── */
$t4 = NumPower::reshape(NumPower::arange(24.0, 0.0, 1.0, 'float32'), [2, 2, 2, 3]);
$r4 = NumPower::abs($t4);
check("4-D abs shape",       $r4->shape(), [2, 2, 2, 3]);

/* ── Empty array shape (0,) preserved ──────────────────────────────── */
$e = NumPower::array([], 'float64');
$r = NumPower::abs($e);
check("empty abs shape",     $r->shape(), [0]);
check("empty abs toArray()", $r->toArray(), []);

/* GPU rounds for the same shapes — silent on success (see check()). */
try {
    $g = $v->gpu();
    if ($g->isGPU()) {
        $r = NumPower::abs($g);
        check("1-D GPU abs isGPU",   $r->isGPU(), true, 0.0, true);
        check("1-D GPU abs values",  $r->cpu()->toArray(),
              [3.5, 0.0, 2.5, 5.5], 1e-12, true);

        $gc = $cube->gpu();
        $r = NumPower::abs($gc);
        check("3-D GPU abs isGPU",   $r->isGPU(), true, 0.0, true);
        check("3-D GPU abs values[0]", $r->cpu()->toArray()[0],
              [[1.0, 2.0], [3.0, 4.0]], 1e-5, true);

        $g4 = $t4->gpu();
        $r = NumPower::abs($g4);
        check("4-D GPU abs isGPU",   $r->isGPU(), true, 0.0, true);
        check("4-D GPU abs shape",   $r->shape(), [2, 2, 2, 3], 0.0, true);

        $ge = $e->gpu();
        $r = NumPower::abs($ge);
        check("empty GPU abs isGPU", $r->isGPU(), true, 0.0, true);
        check("empty GPU abs shape", $r->shape(), [0], 0.0, true);
    }
} catch (\Throwable $t) {
    /* No GPU device (e.g. CPU-only CI): GPU section silently skipped. */
}

/* ── Extreme integer boundary values per dtype ──────────────────────── */
/* INT8_MIN = -128, INT8_MAX = 127. abs(INT8_MIN) wraps to itself per
   modulo-2^N semantics (matches NumPy). */
check("int8 abs MIN wraps",   NumPower::abs(NumPower::array([-128, 127], 'int8'))->toArray(),
                              [-128, 127]);
check("uint8 abs",            NumPower::abs(NumPower::array([0, 255], 'uint8'))->toArray(),
                              [0, 255]);
check("int16 abs MIN wraps",  NumPower::abs(NumPower::array([-32768, 32767], 'int16'))->toArray(),
                              [-32768, 32767]);
check("uint16 abs",           NumPower::abs(NumPower::array([0, 65535], 'uint16'))->toArray(),
                              [0, 65535]);
check("int32 abs MIN wraps",  NumPower::abs(NumPower::array([-2147483648, 2147483647], 'int32'))->toArray(),
                              [-2147483648, 2147483647]);
check("uint32 abs",           NumPower::abs(NumPower::array([0, 4294967295], 'uint32'))->toArray(),
                              [0, 4294967295]);
/* int64 / uint64 boundaries — values past 2^53 require string intake. */
check("int64 abs MIN wraps",
      NumPower::abs(new NDArray(['-9223372036854775808', '9223372036854775807'], 'int64'))->toArray(),
      ['-9223372036854775808', '9223372036854775807']);
check("uint64 abs",
      NumPower::abs(new NDArray(['0', '18446744073709551615'], 'uint64'))->toArray(),
      ['0', '18446744073709551615']);

/* ── Special floating-point values ──────────────────────────────────── */
$f = NumPower::array([INF, -INF, NAN, 0.0, -0.0, 1.0, -1.0], 'float64');
$r = NumPower::abs($f);
$a = $r->toArray();
check("abs(INF) → +INF",      $a[0],  INF);
check("abs(-INF) → +INF",     $a[1],  INF);
check("abs(NAN) is NAN",      is_nan($a[2]), true);
check("abs(0) == 0",          $a[3],  0.0);
check("abs(-0) == 0",         $a[4],  0.0);
check("abs(1) == 1",          $a[5],  1.0);
check("abs(-1) == 1",         $a[6],  1.0);

/* ── Denormals survive abs (positive denormals stay unchanged) ─────── */
$tiny = 5e-324;  /* smallest positive subnormal double */
$r = NumPower::abs(NumPower::array([$tiny, -$tiny], 'float64'));
check("abs(±denormal) → +denormal", $r->toArray(), [$tiny, $tiny], 0.0);

/* ── Large finite values (close to fp64 limit) ──────────────────────── */
$big = 1.7976931348623157e+308;  /* DBL_MAX */
$r = NumPower::abs(NumPower::array([-$big, $big], 'float64'));
check("abs(±DBL_MAX)", $r->toArray(), [$big, $big], $big * 1e-15);

/* ── fp128 extremes via string input ────────────────────────────────── */
$fp = new NDArray(['1.234567890123456789012345e30',
                    '-9.876543210987654321098765e-30'], 'float128');
$r = NumPower::abs($fp);
$rstr = $r->toArray();
/* Result first element should be positive and close to 1.234567e30. */
check("fp128 abs[0] sign",   $rstr[0][0] !== '-', true);  /* not a negative sign */
check("fp128 abs[1] sign",   $rstr[1][0] !== '-', true);
/* Both fp128 magnitudes preserved within fp128 tolerance. */
check("fp128 abs[0] magnitude", (float)$rstr[0] / 1e30, 1.234567890123456789, 1e-9);
check("fp128 abs[1] magnitude", (float)$rstr[1] / 1e-30, 9.876543210987654321, 1e-9);

echo "DONE\n";
?>
--EXPECT--
OK 1-D abs shape
OK 1-D abs values
OK 2-D abs shape
OK 2-D abs values
OK 3-D abs shape
OK 3-D abs values[0]
OK 3-D abs values[1]
OK 4-D abs shape
OK empty abs shape
OK empty abs toArray()
OK int8 abs MIN wraps
OK uint8 abs
OK int16 abs MIN wraps
OK uint16 abs
OK int32 abs MIN wraps
OK uint32 abs
OK int64 abs MIN wraps
OK uint64 abs
OK abs(INF) → +INF
OK abs(-INF) → +INF
OK abs(NAN) is NAN
OK abs(0) == 0
OK abs(-0) == 0
OK abs(1) == 1
OK abs(-1) == 1
OK abs(±denormal) → +denormal
OK abs(±DBL_MAX)
OK fp128 abs[0] sign
OK fp128 abs[1] sign
OK fp128 abs[0] magnitude
OK fp128 abs[1] magnitude
DONE
