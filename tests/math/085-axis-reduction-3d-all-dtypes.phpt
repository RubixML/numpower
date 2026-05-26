--TEST--
3-D axis reduction works for every dtype on CPU and GPU (Rollaxis bug fix)
--FILE--
<?php
/* Pre-existing bug fixed in `NDArray_Rollaxis`: the shift loop wrote one
   element past the allocated `axes` buffer (`axes[n] = axes[n-1]` for
   `n = ndim`) and shifted axes outside the [start..axis] range, producing
   a permutation with a repeated index (e.g. `[1, 0, 1]` for a 3-D
   `rollaxis(a, 1, 0)` instead of `[1, 0, 2]`). `NDArray_Transpose` then
   threw "repeated axis in transpose", which broke 3-D axis-1 reductions
   for every dtype. Also the bound check `!(0 <= start < n + 1)` used a
   broken chained comparison.
   This test covers 3-D shape (2, 2, 3) sum reductions along every axis
   for every dtype on CPU and (when available) GPU. */

$vals = [[[1,2,3], [4,5,6]],
         [[7,8,9], [10,11,12]]];

$expected_axis0 = [[8,10,12], [14,16,18]];  // axis 0 (length 2)
$expected_axis1 = [[5,7,9],   [17,19,21]];  // axis 1 (length 2)
$expected_axis2 = [[6,15],    [24,33]];     // axis 2 (length 3)

$dtypes = [
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
    'float16', 'float32', 'float64', 'float128',
];

$gpu_available = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_available = false; }

function arrays_equal_numeric($a, $b, $eps = 1e-6) {
    if (!is_array($a) || !is_array($b)) return false;
    if (count($a) !== count($b)) return false;
    foreach ($a as $i => $v) {
        if (is_array($v)) {
            if (!arrays_equal_numeric($v, $b[$i], $eps)) return false;
        } else {
            if (abs((float)$v - (float)$b[$i]) > $eps) return false;
        }
    }
    return true;
}

$all_ok = true;
foreach ($dtypes as $dt) {
    $a = NumPower::array($vals, $dt);
    foreach ([[0, $expected_axis0], [1, $expected_axis1], [2, $expected_axis2]] as $case) {
        [$axis, $want] = $case;
        $got_cpu = NumPower::sum($a, $axis)->toArray();
        if (!arrays_equal_numeric($got_cpu, $want)) {
            echo "$dt CPU axis=$axis MISMATCH: got=", json_encode($got_cpu),
                 " want=", json_encode($want), "\n";
            $all_ok = false;
        }
        /* Negative-axis form must give the same result as the positive form. */
        $neg = $axis - 3;
        $got_neg = NumPower::sum($a, $neg)->toArray();
        if (!arrays_equal_numeric($got_neg, $want)) {
            echo "$dt CPU axis=$neg MISMATCH: got=", json_encode($got_neg), "\n";
            $all_ok = false;
        }
        if ($gpu_available) {
            $g = $a->gpu();
            $got_gpu = NumPower::sum($g, $axis)->cpu()->toArray();
            if (!arrays_equal_numeric($got_gpu, $want)) {
                echo "$dt GPU axis=$axis MISMATCH: got=", json_encode($got_gpu), "\n";
                $all_ok = false;
            }
        }
    }
}
echo $all_ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
