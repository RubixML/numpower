--TEST--
NDArray::astype() on a GPU array keeps the array on GPU and converts values
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* When the source array is on the GPU, astype() must keep the result on the
   GPU and convert the values. Cast-back via ->cpu() to inspect values. */

$cases = [
    ['float32', 'int32',   [1.5, 2.5, 3.5], [1, 2, 3]],
    ['int32',   'float64', [1, 2, 3],      [1.0, 2.0, 3.0]],
    ['float64', 'float32', [1.5, 2.5, 0.5], [1.5, 2.5, 0.5]],
    ['int32',   'int16',   [1, 2, 3],      [1, 2, 3]],
];

foreach ($cases as [$src, $dst, $vals, $expect]) {
    $g = (new NDArray($vals, $src))->gpu();
    $r = $g->astype($dst);
    $on_gpu = $r->isGPU();
    $back = $r->cpu()->toArray();
    $ok = ($r instanceof NDArray) && $on_gpu && $back === $expect;
    echo "$src->$dst: ", ($ok ? 'OK' : 'BAD'),
         ' isGPU=', ($r->isGPU() ? 1 : 0),
         ' vals=', json_encode($back), "\n";
}

/* unknown dtype on GPU throws too */
try {
    (new NDArray([1.0], 'float32'))->gpu()->astype('nope');
    echo "badtype: NO-THROW\n";
} catch (Throwable $t) {
    echo "badtype threw: ", get_class($t), "\n";
}
?>
--EXPECT--
float32->int32: OK isGPU=1 vals=[1,2,3]
int32->float64: OK isGPU=1 vals=[1,2,3]
float64->float32: OK isGPU=1 vals=[1.5,2.5,0.5]
int32->int16: OK isGPU=1 vals=[1,2,3]
badtype threw: Error
