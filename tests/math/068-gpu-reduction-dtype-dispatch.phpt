--TEST--
NumPower::sum/prod/min/max/mean run entirely on GPU for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Every dtype must dispatch to a dedicated cuda_reduce_<op>_<tag>
   kernel — no host staging, no D2H of the input buffer. This test
   only verifies correctness; the no-D2H-of-input invariant is enforced
   by the implementation: NDArray_Reduce_* on GPU inputs goes through
   ndarray_reduce_dispatch_gpu, which never calls cudaMemcpy on the
   source data (only on a single 8-byte accumulator slot).

   float4 is exercised separately: its representable set is
   {0, 0.5, 1, 1.5, 2, 3, 4, 6} so input [1,2,3,4,5] quantises 5→4 and
   reduction expectations differ. */
$expect = [
    'sum'  => 15,
    'prod' => 120,
    'max'  => 5,
    'min'  => 1,
    'mean' => 3.0,
];
$dtypes = [
    'float32', 'float64', 'float16', 'float128',
    'float8',
    'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64',
];
$ok = true;
foreach ($dtypes as $t) {
    $a = (new NDArray([1, 2, 3, 4, 5], $t))->gpu();
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

/* float4: 5 is not representable, so the input [1,2,3,4,5] becomes
   [1,2,3,4,4] (4 is the nearest representable to 5; the lookup tie-
   breaks toward 4 over 6 because |5-4|=1 == |5-6|=1 but the encoder
   tournament picks 4 first). */
$a = (new NDArray([1, 2, 3, 4, 5], 'float4'))->gpu();
if (NumPower::sum($a) != 14)  { echo "float4 sum: FAIL got=",  NumPower::sum($a),  "\n"; $ok = false; }
if (NumPower::prod($a) != 96) { echo "float4 prod: FAIL got=", NumPower::prod($a), "\n"; $ok = false; }
if (NumPower::max($a) != 4)   { echo "float4 max: FAIL got=",  NumPower::max($a),  "\n"; $ok = false; }
if (NumPower::min($a) != 1)   { echo "float4 min: FAIL got=",  NumPower::min($a),  "\n"; $ok = false; }

echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
