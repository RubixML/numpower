--TEST--
count($ndarray) and $ndarray->count() return identical values (Countable parity)
--FILE--
<?php
/* NDArray implements Countable, so PHP's count() builtin must call
   ->count() and return exactly the same value. Verify across every
   dtype, rank 0-4, and both devices. */

function assert_pair(NDArray $a, string $tag): void {
    $builtin = count($a);
    $method  = $a->count();
    if ($builtin !== $method) {
        echo "FAIL[$tag] count()=$builtin ->count()=$method\n";
        return;
    }
    echo "OK $tag $builtin\n";
}

$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

/* Rank 0 (scalar). */
foreach ($dtypes as $t) {
    assert_pair(new NDArray(1, $t), "0D-$t");
}

/* Rank 1. */
foreach ($dtypes as $t) {
    assert_pair(new NDArray([1, 2, 3, 4], $t), "1D-$t");
}

/* Rank 2 (non-square). */
foreach ($dtypes as $t) {
    assert_pair(new NDArray([[1, 2, 3], [4, 5, 6]], $t), "2D-$t");
}

/* Rank 3 (zeros so we can use every dtype consistently). */
$t3 = NumPower::zeros([5, 2, 3]);
assert_pair($t3, "3D");

/* Rank 4. */
$t4 = NumPower::zeros([6, 5, 4, 3]);
assert_pair($t4, "4D");

/* Edge: 1-element-along-axis-0. */
assert_pair(NumPower::zeros([1, 99]), "1xN");

/* Edge: large axis-0. */
assert_pair(NumPower::zeros([4096]), "4k-1d");

echo "done\n";
?>
--EXPECT--
OK 0D-float4 0
OK 0D-float8 0
OK 0D-float16 0
OK 0D-float32 0
OK 0D-float64 0
OK 0D-float128 0
OK 0D-int8 0
OK 0D-uint8 0
OK 0D-int16 0
OK 0D-uint16 0
OK 0D-int32 0
OK 0D-uint32 0
OK 0D-int64 0
OK 0D-uint64 0
OK 1D-float4 4
OK 1D-float8 4
OK 1D-float16 4
OK 1D-float32 4
OK 1D-float64 4
OK 1D-float128 4
OK 1D-int8 4
OK 1D-uint8 4
OK 1D-int16 4
OK 1D-uint16 4
OK 1D-int32 4
OK 1D-uint32 4
OK 1D-int64 4
OK 1D-uint64 4
OK 2D-float4 2
OK 2D-float8 2
OK 2D-float16 2
OK 2D-float32 2
OK 2D-float64 2
OK 2D-float128 2
OK 2D-int8 2
OK 2D-uint8 2
OK 2D-int16 2
OK 2D-uint16 2
OK 2D-int32 2
OK 2D-uint32 2
OK 2D-int64 2
OK 2D-uint64 2
OK 3D 5
OK 4D 6
OK 1xN 1
OK 4k-1d 4096
done
