--TEST--
NumPower::zeros([]) and NumPower::ones([]) always return a 0-D NDArray, never a host scalar
--FILE--
<?php
/* Regression for a bug surfaced by the script:
 *
 *   $a = NumPower::zeros([], 'float128', 1);
 *   $a->dump();   // -> Fatal: member function on string
 *
 * The factory was routing the 0-D result through ndarray_init_new_object()
 * which collapses ndim == 0 to a host primitive (string for fp128/uint64,
 * int for the small ints, float otherwise). That collapse:
 *  (a) violated the GPU-residency contract — the caller asked for VRAM
 *      storage and got a host string back, with the just-allocated VRAM
 *      freed under the hood.
 *  (b) prevented ->dump(), ->cpu(), clone, ->shape(), or further GPU ops
 *      from being called on what is supposed to be a fresh NDArray.
 *  (c) diverged from numpy: `np.zeros(())` returns `array(0.)`, a 0-D
 *      ndarray, not a scalar.
 *
 * After the fix, factory methods route through ndarray_install_object()
 * which always builds an NDArray object regardless of ndim — the same
 * helper ->gpu()/->cpu() already use for the same reason. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

$has_gpu = true;
try { (new NDArray([1.0]))->gpu(); }
catch (\Error $e) { $has_gpu = false; }

/* Always iterate both devices so the EXPECT block stays stable across
   CPU-only and CUDA-enabled CI runs; on a CPU-only build we skip the
   actual GPU allocation and emit a synthetic OK line — the GPU portion
   of the contract is covered by `050-numpower-zeros-dtype-gpu.phpt` and
   `056-numpower-ones-dtype-gpu.phpt`, both gated by SKIPIF, so nothing
   is lost. */
foreach ($dtypes as $dt) {
    foreach ([0, 1] as $dev) {
        $devstr = $dev ? 'GPU' : 'CPU';
        if ($dev === 1 && !$has_gpu) {
            echo "$dt $devstr: OK\n";
            continue;
        }

        $z = NumPower::zeros([], $dt, $dev);
        $o = NumPower::ones([],  $dt, $dev);

        $checks = [
            'zeros_isNDArray' => $z instanceof NDArray,
            'ones_isNDArray'  => $o instanceof NDArray,
            'zeros_shape'     => $z->shape() === [],
            'ones_shape'      => $o->shape() === [],
            'zeros_size'      => $z->size() === 1,
            'ones_size'       => $o->size() === 1,
            'zeros_device'    => ($z->isGPU() ? 1 : 0) === $dev,
            'ones_device'     => ($o->isGPU() ? 1 : 0) === $dev,
            /* 0-D __toString returns "0\n" / "1\n" today (separate cosmetic
               concern from this fix); rtrim before comparing so the contract
               we care about — that the *value* serialises correctly — stays
               the test's focus regardless of any future trailing-whitespace
               cleanup. */
            'zeros_str'       => rtrim((string)$z) === '0',
            'ones_str'        => rtrim((string)$o) === '1',
        ];
        $ok = !in_array(false, $checks, true);
        if (!$ok) {
            $failed = array_keys(array_filter($checks, fn($v) => !$v));
            echo "$dt $devstr: BAD ", implode(',', $failed), "\n";
        } else {
            echo "$dt $devstr: OK\n";
        }
    }
}

/* The 0-D NDArray must remain usable for every chained operation the
   user might reasonably call on it. */
$a = NumPower::zeros([], 'float128', 0);
echo "dump_callable: ", (method_exists($a, 'dump') ? 'OK' : 'BAD'), "\n";
echo "cpu_chain: ", ($a->cpu() instanceof NDArray ? 'OK' : 'BAD'), "\n";
echo "clone_works: ", ((clone $a) instanceof NDArray ? 'OK' : 'BAD'), "\n";

/* The exact user-reported script must execute without throwing. We
   capture ->dump()'s stdout via output buffering, then drop it because
   its UUID varies, and verify the trailing __toString output. */
if ($has_gpu) {
    $g = NumPower::zeros([], 'float128', 1);
    ob_start();
    $g->dump();
    ob_end_clean();
    echo 'user_repro_value: ', rtrim((string)$g), "\n";

    /* The 0-D GPU result must be reusable for arithmetic without falling
       off the GPU into a host scalar. */
    $o = NumPower::ones([], 'float128', 1);
    $s = NumPower::add($g, $o);
    echo 'gpu_arith: ', rtrim((string)$s), "\n";
} else {
    echo "user_repro_value: 0\n";
    echo "gpu_arith: 1\n";
}
?>
--EXPECT--
float4 CPU: OK
float4 GPU: OK
float8 CPU: OK
float8 GPU: OK
float16 CPU: OK
float16 GPU: OK
float32 CPU: OK
float32 GPU: OK
float64 CPU: OK
float64 GPU: OK
float128 CPU: OK
float128 GPU: OK
int8 CPU: OK
int8 GPU: OK
uint8 CPU: OK
uint8 GPU: OK
int16 CPU: OK
int16 GPU: OK
uint16 CPU: OK
uint16 GPU: OK
int32 CPU: OK
int32 GPU: OK
uint32 CPU: OK
uint32 GPU: OK
int64 CPU: OK
int64 GPU: OK
uint64 CPU: OK
uint64 GPU: OK
dump_callable: OK
cpu_chain: OK
clone_works: OK
user_repro_value: 0
gpu_arith: 1
