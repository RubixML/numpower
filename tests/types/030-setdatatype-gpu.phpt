--TEST--
NDArray::setDataType() on GPU array stays on GPU and converts values in place
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (Error $e) { die('skip ' . $e->getMessage()); } ?>
--FILE--
<?php
/* When the array is on the GPU, setDataType() must keep it on the GPU
   and convert the values in place (the same object stays on device).
   ->cpu() is used only to read values back — it returns a new array and
   leaves $g on the GPU, so the isGPU() assertion is still meaningful. */

$cases = [
    ['float32', 'int32',   [1.5, 2.5, 3.5], [1, 2, 3]],
    ['int32',   'float64', [1, 2, 3],      [1, 2, 3]],
    ['float64', 'float32', [1.5, 2.5, 0.5], [1.5, 2.5, 0.5]],
    ['int32',   'int16',   [1, 2, 3],      [1, 2, 3]],
];

foreach ($cases as [$src, $dst, $vals, $expect]) {
    $g = (new NDArray($vals, $src))->gpu();
    $pre = $g->dataType();
    $g->setDataType($dst);
    $post = $g->dataType();
    $on_gpu = $g->isGPU();              /* still after setDataType() */
    $back = $g->cpu()->toArray();       /* cpu() returns a new array; $g stays GPU */
    $ok = ($g instanceof NDArray) && $pre === $src && $post === $dst && $on_gpu && $back === $expect;
    echo "$src->$dst: pre=$pre post=$post",
         " isGPU=", ($g->isGPU() ? 1 : 0),
         " vals=", json_encode($back),
         " ok=", ($ok ? 'OK' : 'BAD'), "\n";
}

/* unknown dtype on GPU throws too, leaving the array intact on GPU */
$g2 = (new NDArray([1.0, 2.0], 'float32'))->gpu();
$pre2 = $g2->dataType();
try {
    $g2->setDataType('nope');
    echo "badtype: NO-THROW\n";
} catch (Throwable $t) {
    echo "badtype: pre=$pre2 post=", $g2->dataType(),
         " isGPU=", ($g2->isGPU() ? 1 : 0),
         " vals=", json_encode($g2->cpu()->toArray()), "\n";
}
?>
--EXPECT--
float32->int32: pre=float32 post=int32 isGPU=1 vals=[1,2,3] ok=OK
int32->float64: pre=int32 post=float64 isGPU=1 vals=[1,2,3] ok=OK
float64->float32: pre=float64 post=float32 isGPU=1 vals=[1.5,2.5,0.5] ok=OK
int32->int16: pre=int32 post=int16 isGPU=1 vals=[1,2,3] ok=OK
badtype: pre=float32 post=float32 isGPU=1 vals=[1,2]
