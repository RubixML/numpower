--TEST--
NumPower unary ops on float4 / float8 (narrow non-half floats): compute through float32, cast back, stay on device
--FILE--
<?php
/* float4 (E2M1, 16 representable values) and float8 (E4M3, ~240 max value)
   have no native intrinsics on either CPU or GPU. The unary dispatcher
   routes them through float32 for compute, then casts back to the source
   dtype. On GPU the cast stays on-device (cuda_cast_fp{4,8}_to_f32 /
   f32_to_fp{4,8}). This test exercises that path. */

$gpu_available = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_available = false; }

function near($a, $b, $tol = 1e-5) {
    return abs((float)$a - (float)$b) <= $tol;
}

function check_arr($label, $got, $want, $tol = 1e-5) {
    if (count($got) !== count($want)) {
        echo "FAIL $label: len ", count($got), " vs ", count($want), "\n";
        return;
    }
    foreach ($got as $i => $g) {
        if (!near($g, $want[$i], $tol)) {
            echo "FAIL $label[$i]: got=$g want={$want[$i]}\n";
            return;
        }
    }
    echo "OK $label\n";
}

/* fp4 representable: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
   Use only representable values so the cast round-trip is exact. */
$fp4_in = NumPower::array([-3.0, -1.5, 0.0, 1.0, 2.0, 4.0], 'float4');
check_arr('fp4 abs (CPU)',      NumPower::abs($fp4_in)->toArray(),
                                 [3.0, 1.5, 0.0, 1.0, 2.0, 4.0]);
check_arr('fp4 negative (CPU)', NumPower::negative($fp4_in)->toArray(),
                                 [3.0, 1.5, 0.0, -1.0, -2.0, -4.0]);
check_arr('fp4 sign (CPU)',     NumPower::sign($fp4_in)->toArray(),
                                 [-1.0, -1.0, 0.0, 1.0, 1.0, 1.0]);
check_arr('fp4 square (CPU)',   NumPower::square(NumPower::array([1.0, 2.0], 'float4'))->toArray(),
                                 [1.0, 4.0]);
check_arr('fp4 sqrt (CPU)',     NumPower::sqrt(NumPower::array([1.0, 4.0], 'float4'))->toArray(),
                                 [1.0, 2.0]);
check_arr('fp4 clip (CPU)',     NumPower::clip(NumPower::array([-6.0, 0.0, 6.0], 'float4'), -2.0, 2.0)->toArray(),
                                 [-2.0, 0.0, 2.0]);

/* fp8 (E4M3) has finer granularity than fp4 but limited precision. */
$fp8_in = NumPower::array([-2.0, -0.5, 0.0, 1.0, 3.0], 'float8');
check_arr('fp8 abs (CPU)',      NumPower::abs($fp8_in)->toArray(),
                                 [2.0, 0.5, 0.0, 1.0, 3.0]);
check_arr('fp8 negative (CPU)', NumPower::negative($fp8_in)->toArray(),
                                 [2.0, 0.5, 0.0, -1.0, -3.0], 0.1);
check_arr('fp8 sqrt (CPU)',     NumPower::sqrt(NumPower::array([1.0, 4.0, 16.0], 'float8'))->toArray(),
                                 [1.0, 2.0, 4.0]);
check_arr('fp8 reciprocal (CPU)', NumPower::reciprocal(NumPower::array([1.0, 2.0, 4.0], 'float8'))->toArray(),
                                 [1.0, 0.5, 0.25], 0.05);
check_arr('fp8 clip (CPU)',     NumPower::clip(NumPower::array([-100.0, 0.0, 100.0], 'float8'), -10.0, 10.0)->toArray(),
                                 [-10.0, 0.0, 10.0]);

if (!$gpu_available) {
    echo "SKIP_GPU\n";
    echo "DONE\n";
    return;
}

/* GPU: cast must stay on-device. */
$g4 = NumPower::array([-1.5, 0.0, 1.0, 2.0], 'float4')->gpu();
$r4 = NumPower::abs($g4);
if (!$r4->isGPU()) { echo "FAIL fp4 abs left GPU\n"; }
else               { check_arr('fp4 abs (GPU)', $r4->cpu()->toArray(), [1.5, 0.0, 1.0, 2.0]); }

$g8 = NumPower::array([-2.0, 0.0, 1.0, 3.0], 'float8')->gpu();
$r8 = NumPower::sqrt(NumPower::abs($g8));
if (!$r8->isGPU()) { echo "FAIL fp8 sqrt left GPU\n"; }
else               { check_arr('fp8 sqrt(abs) (GPU)', $r8->cpu()->toArray(),
                                                       [sqrt(2.0), 0.0, 1.0, sqrt(3.0)], 0.1); }

/* fp4 clip on GPU. */
$g4c = NumPower::array([-6.0, 0.0, 6.0], 'float4')->gpu();
$r4c = NumPower::clip($g4c, -2.0, 2.0);
if (!$r4c->isGPU()) { echo "FAIL fp4 clip left GPU\n"; }
else                { check_arr('fp4 clip (GPU)', $r4c->cpu()->toArray(), [-2.0, 0.0, 2.0]); }

echo "DONE\n";
?>
--EXPECTF--
OK fp4 abs (CPU)
OK fp4 negative (CPU)
OK fp4 sign (CPU)
OK fp4 square (CPU)
OK fp4 sqrt (CPU)
OK fp4 clip (CPU)
OK fp8 abs (CPU)
OK fp8 negative (CPU)
OK fp8 sqrt (CPU)
OK fp8 reciprocal (CPU)
OK fp8 clip (CPU)
%A
DONE
