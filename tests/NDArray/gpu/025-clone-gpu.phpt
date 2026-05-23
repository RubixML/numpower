--TEST--
clone $ndarray on GPU produces a working GPU NDArray
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Reported by the user: the chained form
       $np = (new NDArray(1, 'float128'))->gpu();
       $x  = clone $np;
       echo $x + 2;
   was throwing "Unsupported operand types: NDArray + int". The custom
   clone handler now wires the do_operation slot and deep-copies the
   GPU buffer via cudaMemcpy DeviceToDevice — both fixes verified here. */

/* The exact user reproducer. */
$np = (new NDArray(1, 'float128'))->gpu();
$x  = clone $np;
$r  = $x + 2;
$ok1 = $x instanceof NDArray && $x->isGPU() === 1;
echo "user case: ", ($ok1 && (string)$r === "3") ? "ok\n" : "FAIL r=" . var_export($r, true) . "\n";

/* Clone of a GPU 1-D array — arithmetic must stay on GPU and produce
   correct values. */
$a = (new NDArray([1.0, 2.0, 3.0], 'float32'))->gpu();
$b = clone $a;
$ok2 = $b instanceof NDArray && $b->isGPU() === 1
    && ($b + 1.0)->cpu()->toArray() === [2.0, 3.0, 4.0];
echo "GPU 1-D + scalar: ", $ok2 ? "ok\n" : "FAIL\n";

/* Deep-copy semantics on GPU: clone, move clone to CPU, source must
   stay on GPU and unchanged. */
$s  = (new NDArray([10.0, 20.0], 'float32'))->gpu();
$sc = clone $s;
$sc_cpu = $sc->cpu();
$ok3 = $s->isGPU() === 1                    /* source still on GPU */
    && $s->cpu()->toArray() === [10.0, 20.0]/* source values intact */
    && $sc_cpu->toArray() === [10.0, 20.0]; /* clone's values intact */
echo "deep copy: ", $ok3 ? "ok\n" : "FAIL\n";

/* Mix of dtypes — clone must work for every native + emulated dtype. */
$dtypes = [
    'float32', 'float64', 'float16', 'float128', 'float4', 'float8',
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
];
$all_ok = true;
foreach ($dtypes as $t) {
    $g = (new NDArray([1, 2, 3, 4, 5], $t))->gpu();
    $c = clone $g;
    if (!($c instanceof NDArray) || $c->isGPU() !== 1) {
        echo "$t: FAIL not-NDArray-or-not-GPU\n";
        $all_ok = false;
    }
}
echo "all dtypes: ", $all_ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
user case: ok
GPU 1-D + scalar: ok
deep copy: ok
all dtypes: ok
