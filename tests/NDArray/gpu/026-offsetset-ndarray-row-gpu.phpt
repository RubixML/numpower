--TEST--
NDArray::offsetSet() — `$a[i] = NDArray` on a GPU array stays on GPU, byte-correct
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Non-scalar offsetSet on a GPU array routes through NDArray_Overwrite which,
   for same-dtype same-device pairs, uses a single vmemcpyd2d() with no host
   round-trip. The result must be identical to the CPU path and keep the
   destination on the GPU. */

$dtypes = ['float32', 'float64', 'int32', 'int64'];

foreach ($dtypes as $t) {
    $a = (new NDArray([[1, 2, 3], [4, 5, 6]], $t))->gpu();
    $row = (new NDArray([70, 80, 90], $t))->gpu();
    $a[0] = $row;

    $on_gpu = $a->isGPU();
    /* Read back through CPU to verify byte-correctness. */
    $back = $a->cpu()->toArray();
    $ok = $back[0] == [70, 80, 90] && $back[1] == [4, 5, 6];

    echo "$t: gpu=", $on_gpu ? "yes" : "no", " row=", $ok ? "OK" : "BAD", "\n";
}

/* Mixed-device assignment must be rejected. */
$b = (new NDArray([[1, 2], [3, 4]], 'float32'))->gpu();
$cpu_src = new NDArray([7.0, 8.0], 'float32');
$err = 'NONE';
try { $b[0] = $cpu_src; } catch (\Throwable $e) { $err = 'THROWN'; }
echo "mixed-device: $err\n";
?>
--EXPECT--
float32: gpu=yes row=OK
float64: gpu=yes row=OK
int32: gpu=yes row=OK
int64: gpu=yes row=OK
mixed-device: THROWN
