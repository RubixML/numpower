--TEST--
4-D axis reduction (Rollaxis bug fix) + empty-axis GPU output + symmetry of string-scalar arithmetic
--FILE--
<?php
/* Higher-rank axis reduction stresses the Rollaxis shift loop further than
   the 3-D coverage in 085: with ndim==4 the loop has to shift two axes
   when reducing along axis 1 or 2, and the negative-axis form has to land
   on the same destination as the matching positive index in [-ndim, ndim).
   The same code path also has to honour the empty-axis identity-fill on
   both CPU and GPU for every dtype. */

$gpu_ok = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_ok = false; }

$ok = true;

/* 4-D axis reduction across every axis (positive + negative). */
$a = NumPower::array([
    [[[1,2],[3,4]],[[5,6],[7,8]]],
    [[[9,10],[11,12]],[[13,14],[15,16]]]
], 'int32');
$want = [
    0 => [[[10,12],[14,16]],[[18,20],[22,24]]],
    1 => [[[6,8],[10,12]],[[22,24],[26,28]]],
    2 => [[[4,6],[12,14]],[[20,22],[28,30]]],
    3 => [[[3,7],[11,15]],[[19,23],[27,31]]],
];
foreach ([0,1,2,3] as $axis) {
    $r = NumPower::sum($a, $axis)->toArray();
    if ($r !== $want[$axis]) {
        echo "FAIL 4D CPU axis=$axis: ", json_encode($r), "\n";
        $ok = false;
    }
    /* Negative-axis form. */
    $neg = $axis - 4;
    $r_neg = NumPower::sum($a, $neg)->toArray();
    if ($r_neg !== $want[$axis]) {
        echo "FAIL 4D CPU axis=$neg: ", json_encode($r_neg), "\n";
        $ok = false;
    }
    if ($gpu_ok) {
        $r_gpu = NumPower::sum($a->gpu(), $axis)->cpu()->toArray();
        if ($r_gpu !== $want[$axis]) {
            echo "FAIL 4D GPU axis=$axis: ", json_encode($r_gpu), "\n";
            $ok = false;
        }
    }
}

/* Empty-axis identity-fill: shape (0,3,2) along axis=0 → fill of (3,2).
   Use json_encode round-trip so the comparison ignores int/float/string
   storage differences across dtypes (uint64 / float128 land in strings,
   the rest in numbers). */
function as_int_grid($a) {
    return array_map(fn($row) => array_map(fn($v) => (int)$v, $row), $a);
}
$want_zero = [[0,0],[0,0],[0,0]];
$want_one  = [[1,1],[1,1],[1,1]];
foreach (['int8','int32','int64','uint8','uint32','uint64','float32','float64','float128'] as $dt) {
    $empty = NumPower::zeros([0, 3, 2], $dt);
    $r0 = as_int_grid(NumPower::sum($empty, 0)->toArray());
    if ($r0 !== $want_zero) {
        echo "FAIL $dt empty axis=0 sum: ", json_encode($r0), "\n";
        $ok = false;
    }
    $r1 = as_int_grid(NumPower::prod($empty, 0)->toArray());
    if ($r1 !== $want_one) {
        echo "FAIL $dt empty axis=0 prod: ", json_encode($r1), "\n";
        $ok = false;
    }
    if ($gpu_ok) {
        $eg = $empty->gpu();
        $r0g = NumPower::sum($eg, 0);
        if (!$r0g->isGPU()) {
            echo "FAIL $dt empty GPU sum not on GPU\n";
            $ok = false;
        }
        $r0gcpu = as_int_grid($r0g->cpu()->toArray());
        if ($r0gcpu !== $want_zero) {
            echo "FAIL $dt empty GPU axis=0 sum: ", json_encode($r0gcpu), "\n";
            $ok = false;
        }
        $r1g = as_int_grid(NumPower::prod($eg, 0)->cpu()->toArray());
        if ($r1g !== $want_one) {
            echo "FAIL $dt empty GPU axis=0 prod: ", json_encode($r1g), "\n";
            $ok = false;
        }
    }
}

/* String-scalar symmetry: `add(arr, '5')` and `add('5', arr)` must produce
   the same NDArray for every dtype, including the wide-precision ones. */
foreach (['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
          'float16', 'float32', 'float64', 'float128'] as $dt) {
    $arr = new NDArray([10, 20, 30], $dt);
    $r1 = NumPower::add($arr, '5')->toArray();
    $r2 = NumPower::add('5', $arr)->toArray();
    if ($r1 !== $r2) {
        echo "FAIL $dt scalar string asymmetric: $r1 vs $r2\n";
        $ok = false;
    }
}

/* Symmetric subtract is direction-sensitive: a-5 != 5-a, but both should
   produce a valid NDArray with the source's dtype. */
foreach (['int32', 'int64', 'uint32'] as $dt) {
    $arr = new NDArray([10, 20, 30], $dt);
    $diff = NumPower::subtract($arr, '5')->toArray();
    $diff_rev = NumPower::subtract('5', $arr)->toArray();
    if (count($diff) !== 3 || count($diff_rev) !== 3) {
        echo "FAIL $dt subtract shape\n";
        $ok = false;
    }
}

echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
