--TEST--
GPU linalg ops (det/svd/inv/norm) do not leak VRAM
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Pre-existing leak bugs fixed in May 2026:
   - cuda_svd_float    : CHECK_* macros bypassed cleanup of handles/buffers
   - cuda_det_float    : passed NULL workspace to cusolverDnSgetrf (garbage
                          output AND corrupted device state on next vmalloc)
   - cuda_matrix_float_inverse  : ignored every cusolverDn return code; no
                                  cleanup on failure
   - cuda_matrix_float_l2norm   : CHECK_* macros leaked handles/d_work on
                                  early failure
   All four now route every failure through a single cleanup label that
   releases every allocated resource. This test exercises the happy path
   in a loop; VCHECK at RSHUTDOWN catches any vmalloc/vfree imbalance. */
for ($i = 0; $i < 20; $i++) {
    /* det */
    $m = (new NDArray([[1.0, 2.0], [3.0, 4.0]], 'float32'))->gpu();
    $d = NumPower::det($m);

    /* svd (3x2 — full SVD path) */
    $a = (new NDArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 'float32'))->gpu();
    $svd = NumPower::svd($a);

    /* inverse */
    $b = (new NDArray([[4.0, 7.0], [2.0, 6.0]], 'float32'))->gpu();
    $inv = NumPower::inv($b);

    /* l2 norm */
    $c = (new NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 'float32'))->gpu();
    $nrm = NumPower::norm($c);

    unset($m, $d, $a, $svd, $b, $inv, $c, $nrm);
}
echo "ok\n";
?>
--EXPECT--
ok
