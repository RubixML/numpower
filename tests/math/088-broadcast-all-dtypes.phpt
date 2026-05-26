--TEST--
NDArray_Broadcast covers every dtype (1-D→2-D, column / row, scalar → N-D)
--FILE--
<?php
/* Pre-existing bug fixed: NDArray_Broadcast hardcoded `sizeof(float)` for
   the float32 branch and `sizeof(double)` for everything else, silently
   reading/writing 4–7 bytes past the source for narrow ints (`int8`,
   `int16`, `int32`, `uint8`..`uint32`). The buggy code was reached only
   when the dispatcher routed a non-float dtype through the broadcast,
   which happened the moment the native int CPU kernel started running
   for those dtypes.
   The fix uses `NDArray_ELSIZE` throughout so memcpy byte counts match
   the actual storage width. */

$dtypes = [
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
    'float16', 'float32', 'float64', 'float128',
];
$gpu_available = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_available = false; }

function normalize($a) {
    /* Cast nested int/string values to int for comparison so uint64 /
       float128 strings line up with the int dtypes' integer values. */
    if (is_array($a)) {
        return array_map(fn ($v) => normalize($v), $a);
    }
    return (int)$a;
}

$all_ok = true;

/* 1-D + 2-D row broadcast: b broadcast across rows of a. */
$expect_row = [[11, 22, 33], [14, 25, 36]];
foreach ($dtypes as $dt) {
    $a = NumPower::array([[1, 2, 3], [4, 5, 6]], $dt);
    $b = NumPower::array([10, 20, 30], $dt);
    $r = NumPower::add($a, $b);
    if (normalize($r->toArray()) !== $expect_row) {
        echo "FAIL $dt row-broadcast CPU: ", json_encode($r->toArray()), "\n";
        $all_ok = false;
    }
    if ($gpu_available) {
        $gr = NumPower::add($a->gpu(), $b->gpu());
        if (!$gr->isGPU()) { echo "FAIL $dt row-broadcast GPU not on GPU\n"; $all_ok = false; }
        if (normalize($gr->cpu()->toArray()) !== $expect_row) {
            echo "FAIL $dt row-broadcast GPU: ", json_encode($gr->cpu()->toArray()), "\n";
            $all_ok = false;
        }
    }
}

/* 2-D column broadcast: b shape (R,1) broadcast across columns of a (R,C). */
$expect_col = [[11, 12, 13], [24, 25, 26]];
foreach ($dtypes as $dt) {
    $a = NumPower::array([[1, 2, 3], [4, 5, 6]], $dt);
    $b = NumPower::array([[10], [20]], $dt);
    $r = NumPower::add($a, $b);
    if (normalize($r->toArray()) !== $expect_col) {
        echo "FAIL $dt col-broadcast CPU: ", json_encode($r->toArray()), "\n";
        $all_ok = false;
    }
    if ($gpu_available) {
        $gr = NumPower::add($a->gpu(), $b->gpu());
        if (normalize($gr->cpu()->toArray()) !== $expect_col) {
            echo "FAIL $dt col-broadcast GPU: ", json_encode($gr->cpu()->toArray()), "\n";
            $all_ok = false;
        }
    }
}

/* Scalar (0-D NDArray) broadcast to 2-D. */
$expect_scalar = [[11, 12, 13], [14, 15, 16]];
foreach ($dtypes as $dt) {
    $a = NumPower::array([[1, 2, 3], [4, 5, 6]], $dt);
    $b = new NDArray(10, $dt);  /* 0-D */
    $r = NumPower::add($a, $b);
    if (normalize($r->toArray()) !== $expect_scalar) {
        echo "FAIL $dt scalar-broadcast CPU: ", json_encode($r->toArray()), "\n";
        $all_ok = false;
    }
}

/* 3-D + 1-D: row broadcast across last dim. */
$expect_3d = [[[11, 22, 33], [14, 25, 36]],
              [[17, 28, 39], [20, 31, 42]]];
foreach ($dtypes as $dt) {
    $a = NumPower::array([[[1, 2, 3], [4, 5, 6]],
                          [[7, 8, 9], [10, 11, 12]]], $dt);
    $b = NumPower::array([10, 20, 30], $dt);
    $r = NumPower::add($a, $b);
    if (normalize($r->toArray()) !== $expect_3d) {
        echo "FAIL $dt 3D+1D CPU: ", json_encode($r->toArray()), "\n";
        $all_ok = false;
    }
}

echo $all_ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
