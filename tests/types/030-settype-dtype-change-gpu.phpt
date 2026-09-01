--TEST--
NDArray::setType() on a GPU array stays on GPU and converts values in place
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* When the array is on the GPU, setType() must keep it on the GPU and
   convert the values in place (the same object stays on device).
   ->cpu() is used only to read back values — it returns a new array and
   leaves $g on the GPU, so the isGPU() assertion is still meaningful. */

$cases = [
    ['float32', 'int32',   [1.5, 2.5, 3.5], [1, 2, 3]],
    ['int32',   'float64', [1, 2, 3],      [1.0, 2.0, 3.0]],
    ['float64', 'float32', [1.5, 2.5, 0.5], [1.5, 2.5, 0.5]],
    ['int32',   'int16',   [1, 2, 3],      [1, 2, 3]],
];

foreach ($cases as [$src, $dst, $vals, $expect]) {
    $g = (new NDArray($vals, $src))->gpu();
    $g->setType($dst);
    $on_gpu = $g->isGPU();          /* still after setType() */
    $back = $g->cpu()->toArray();   /* cpu() returns a new array; $g stays GPU */
    $ok = ($g instanceof NDArray) && $on_gpu && $back === $expect;
    echo "$src->$dst: ", ($ok ? 'OK' : 'BAD'),
         ' isGPU=', ($g->isGPU() ? 1 : 0),
         ' vals=', json_encode($back), "\n";
}

/* unknown dtype on GPU throws too, leaving the array intact on GPU */
$g2 = (new NDArray([1.0, 2.0], 'float32'))->gpu();
try {
    $g2->setType('nope');
    echo "badtype: NO-THROW\n";
} catch (Throwable $t) {
    echo "badtype threw: ", get_class($t),
         " isGPU=", ($g2->isGPU() ? 1 : 0),
         " vals=", json_encode($g2->cpu()->toArray()), "\n";
}
?>
--EXPECT--
float32->int32: OK isGPU=1 vals=[1,2,3]
int32->float64: OK isGPU=1 vals=[1,2,3]
float64->float32: OK isGPU=1 vals=[1.5,2.5,0.5]
int32->int16: OK isGPU=1 vals=[1,2,3]
badtype threw: Error isGPU=1 vals=[1.0,2.0]
