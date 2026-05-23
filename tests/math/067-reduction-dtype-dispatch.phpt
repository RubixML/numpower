--TEST--
NumPower::sum/prod/min/max/mean dispatch correctly for every dtype (CPU)
--FILE--
<?php
/* Before this fix every reduction routed through NDArray_Sum_Float /
   NDArray_Float_Prod / NDArray_Min / NDArray_Max, which cast the buffer
   to (float*) regardless of dtype — float64, int*, uint* inputs all
   returned garbage. The new NDArray_Reduce_* helpers dispatch by dtype
   string and read each element via ndarray_element_to_double. */
$values = [1, 2, 3, 4, 5];
$expect = [
    'sum'  => 15,
    'prod' => 120,
    'max'  => 5,
    'min'  => 1,
    'mean' => 3.0,
];

/* float4/float8/float16 are reduced-precision and lose accuracy on
   integers near the limit; cover them separately. */
$dtypes = [
    'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
];

$ok = true;
foreach ($dtypes as $t) {
    $a = new NDArray($values, $t);
    $got = [
        'sum'  => NumPower::sum($a),
        'prod' => NumPower::prod($a),
        'max'  => NumPower::max($a),
        'min'  => NumPower::min($a),
        'mean' => NumPower::mean($a),
    ];
    foreach ($expect as $op => $want) {
        $g = (float)$got[$op];
        if (abs($g - $want) > 1e-9) {
            echo "$t $op: FAIL want=$want got=$g\n";
            $ok = false;
        }
    }
}
echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
