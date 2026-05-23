--TEST--
SVD and `==` on GPU NDArrays route their device scratch through vmalloc (visible to NDARRAY_VCHECK)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Before the fix these helpers in src/ndmath/cuda/cuda_math.cu used raw
 * `cudaMalloc` / `cudaFree`, bypassing the NDARRAY_VCHECK counter. A real
 * VRAM leak in svd's workspace or array-equals's flag slot would be
 * invisible. Run each path many times inside a function so the scope-end
 * destruction balances the counter; if either op leaks even one byte the
 * RSHUTDOWN check will print a non-zero leak count and this test fails. */

function svd_rounds(): void {
    for ($i = 0; $i < 32; $i++) {
        $a = NumPower::array(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]],
            'float32'
        )->gpu();
        $svd = NumPower::svd($a);  /* [U, S, V] — uses cuda_svd_float scratch */
        unset($a, $svd);
    }
}

function equal_rounds(): void {
    for ($i = 0; $i < 32; $i++) {
        $a = NumPower::array([1., 2., 3., 4.], 'float32')->gpu();
        $b = NumPower::array([1., 2., 3., 4.], 'float32')->gpu();
        $c = NumPower::array([1., 2., 3., 5.], 'float32')->gpu();
        /* `==` between two NDArray objects funnels through compare_ndarrays,
           which on the GPU path calls cuda_equal_float. */
        if ($a == $b) {
            /* ok */
        }
        if ($a == $c) {
            echo "mismatched compared equal\n";
        }
        unset($a, $b, $c);
    }
}

svd_rounds();
equal_rounds();
echo "ok\n";
?>
--EXPECT--
ok
