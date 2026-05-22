--TEST--
NumPower::slice() GPU strided-copy kernel: negative strides, 3-D+ patterns, all elsize fast-paths
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Exercises the GPU strided-copy kernel introduced for NDArray_Slice.
   Each case covers a path the per-row cudaMemcpy loop handled poorly or not
   at all:
   - negative strides (slice with step < 0) — the kernel multiplies idx by
     a signed long-long stride, no host-side reversal needed.
   - 3-D strided slice — kernel decomposes the linear thread id into a
     multi-index of arbitrary rank in one launch.
   - elsize 1, 2, 4, 8, 16: each takes a different fast path in the kernel's
     switch on elsize. */

function compare(NDArray $cpu, NDArray $gpu): bool {
    return $cpu->shape() === $gpu->shape() && $cpu->toArray() === $gpu->toArray();
}

/* ---------- negative-step slicing on GPU ---------- */
$src = NumPower::arange(20.0);
echo "1D negstep step=-1: ",
     compare(NumPower::slice($src,[19, -1, -1]),                  /* reverse the whole array */
             NumPower::slice($src->gpu(),[19, -1, -1])->cpu()) ? "OK" : "BAD", "\n";
echo "1D negstep step=-3: ",
     compare(NumPower::slice($src,[18, 0, -3]),
             NumPower::slice($src->gpu(),[18, 0, -3])->cpu()) ? "OK" : "BAD", "\n";

/* 2D negative-step on both axes. To include index 0 with a negative step
   the stop has to land strictly below 0 — we use -len-1 as a clear sentinel.
   axis 0 (len 4): [3, -5, -1] → indices 3, 2, 1, 0
   axis 1 (len 6): [5, -7, -2] → indices 5, 3, 1 */
$m = NumPower::array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]], 'float32');
$cpu_2 = NumPower::slice($m, [3, -5, -1], [5, -7, -2]);
$gpu_2 = NumPower::slice($m->gpu(), [3, -5, -1], [5, -7, -2])->cpu();
echo "2D negstep both axes: ", compare($cpu_2, $gpu_2) ? "OK" : "BAD", "\n";
echo "2D negstep shape: ", json_encode($cpu_2->shape()), "\n";
echo "2D negstep values: ", json_encode($cpu_2->toArray()), "\n";

/* ---------- 3-D slicing on GPU through the kernel ---------- */
$data = [];
$k = 0;
for ($i = 0; $i < 4; $i++) {
    $data[$i] = [];
    for ($j = 0; $j < 5; $j++) {
        $data[$i][$j] = [];
        for ($l = 0; $l < 6; $l++) $data[$i][$j][$l] = (float)$k++;
    }
}
$cube_cpu = new NDArray($data, 'float32');
$cube_gpu = $cube_cpu->gpu();

/* Each axis kept, then mixed: kernel hits ndim ∈ {2, 3}. */
echo "3D slice([],[],-1): ",
     compare(NumPower::slice($cube_cpu, [], [], -1),
             NumPower::slice($cube_gpu, [], [], -1)->cpu()) ? "OK" : "BAD", "\n";
echo "3D slice([1,3],[0,5,2],[5,0,-1]): ",
     compare(NumPower::slice($cube_cpu, [1, 3], [0, 5, 2], [5, 0, -1]),
             NumPower::slice($cube_gpu, [1, 3], [0, 5, 2], [5, 0, -1])->cpu()) ? "OK" : "BAD", "\n";
echo "3D slice([2,-1,-1],[],[]): ",
     compare(NumPower::slice($cube_cpu, [2, -1, -1], [], []),
             NumPower::slice($cube_gpu, [2, -1, -1], [], [])->cpu()) ? "OK" : "BAD", "\n";

/* ---------- elsize coverage: 1, 2, 4, 8, 16 byte fast paths ---------- */
$dtype_by_size = [
    1  => 'int8',
    2  => 'int16',
    4  => 'int32',
    8  => 'int64',
    16 => 'float128',
];
foreach ($dtype_by_size as $sz => $t) {
    /* Strided column slice — uses the kernel's `elsize==$sz` switch arm. */
    $m = new NDArray([[1,2,3,4],[5,6,7,8],[9,10,11,12]], $t);
    $cpu_col = NumPower::slice($m, [], -2);
    $gpu_col = NumPower::slice($m->gpu(), [], -2)->cpu();
    echo "$t (elsize=$sz) col: ", compare($cpu_col, $gpu_col) ? "OK" : "BAD", "\n";

    /* Negative-step row slice */
    $cpu_rev = NumPower::slice($m, [2, -1, -1]);
    $gpu_rev = NumPower::slice($m->gpu(), [2, -1, -1])->cpu();
    echo "$t (elsize=$sz) rev: ", compare($cpu_rev, $gpu_rev) ? "OK" : "BAD", "\n";
}

/* ---------- view_shape with a 1-length axis kept (kernel hits shape[i]==1) ---------- */
$m  = NumPower::array([[[1,2,3],[4,5,6]]], 'float32');   /* shape [1,2,3] */
$ok = compare(NumPower::slice($m, [], [0, 2], [1, 3]),
              NumPower::slice($m->gpu(), [], [0, 2], [1, 3])->cpu());
echo "shape-1 leading axis: ", $ok ? "OK" : "BAD", "\n";
?>
--EXPECT--
1D negstep step=-1: OK
1D negstep step=-3: OK
2D negstep both axes: OK
2D negstep shape: [4,3]
2D negstep values: [[24,22,20],[18,16,14],[12,10,8],[6,4,2]]
3D slice([],[],-1): OK
3D slice([1,3],[0,5,2],[5,0,-1]): OK
3D slice([2,-1,-1],[],[]): OK
int8 (elsize=1) col: OK
int8 (elsize=1) rev: OK
int16 (elsize=2) col: OK
int16 (elsize=2) rev: OK
int32 (elsize=4) col: OK
int32 (elsize=4) rev: OK
int64 (elsize=8) col: OK
int64 (elsize=8) rev: OK
float128 (elsize=16) col: OK
float128 (elsize=16) rev: OK
shape-1 leading axis: OK
