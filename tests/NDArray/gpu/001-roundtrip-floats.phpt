--TEST--
NDArray::gpu()/cpu() preserves data for all float dtypes
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Round-trip for every float dtype: CPU -> GPU -> CPU must yield identical
   element-for-element output. We use string-typed inputs for the exotic
   reduced-precision types so the values are exactly representable. */
$cases = [
    'float4'   => ['0', '0.5', '1', '1.5', '2', '3', '4', '6'],
    'float8'   => ['0', '0.5', '1', '1.5', '2', '3', '4', '6'],
    'float16'  => ['0', '0.5', '1', '1.5', '-0.5', '-1', '-2', '-6'],
    'float32'  => [0.0, 1.0, 0.5, 1.5, -0.5, -1.5, 1e-3, 1e3],
    'float64'  => [0.0, 1.0, 0.5, 1.5, -0.5, -1.5, 1e-9, 1e9],
    /* fp128 on GPU is stored as double-double; only values exactly
       representable in fp64 (integers below 2^53 and simple binary fractions)
       round-trip bit-exact. Non-exact values like 1e-9, 1e-300, 1e308 lose
       ~7 bits in the dd encoding. */
    'float128' => ['0', '1', '0.5', '1.5', '-0.5', '-1.5', '1000000000', '0.0625'],
];

foreach ($cases as $dtype => $values) {
    $cpu = new NDArray($values, $dtype);
    $before = (string)$cpu;
    $gpu = $cpu->gpu();
    if (!$gpu->isGPU()) {
        echo "$dtype: gpu() did not move to GPU\n";
        continue;
    }
    $back = $gpu->cpu();
    if ($back->isGPU()) {
        echo "$dtype: cpu() did not move to CPU\n";
        continue;
    }
    $after = (string)$back;
    echo $dtype, ': ', ($before === $after ? 'OK' : "DIFF before=$before after=$after"), "\n";
}
?>
--EXPECT--
float4: OK
float8: OK
float16: OK
float32: OK
float64: OK
float128: OK
