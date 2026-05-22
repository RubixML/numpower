--TEST--
NDArray::count() exactly matches the number of foreach iterations (Iterator+Countable parity)
--FILE--
<?php
/* Iterator and Countable describe the SAME axis-0 traversal, so count()
   must equal the number of values foreach yields -- on every dtype, every
   rank, and both devices. A drift between them breaks the standard PHP
   idiom of pre-allocating a result buffer with count() before a foreach
   loop. */

function assert_iter_parity(NDArray $a, string $tag): void {
    $expected = $a->count();
    $iters = 0;
    foreach ($a as $sub) {
        $iters++;
        if ($iters > 10000) {
            echo "FAIL $tag: runaway loop\n";
            return;
        }
    }
    if ($iters !== $expected) {
        echo "FAIL $tag: count()=$expected foreach=$iters\n";
        return;
    }
    echo "OK $tag $expected\n";
}

$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

/* 1-D: foreach yields scalars. */
foreach ($dtypes as $t) {
    assert_iter_parity(new NDArray([1, 2, 3, 4, 5, 6, 7], $t), "1D-$t");
}

/* 2-D: foreach yields rows. */
foreach ($dtypes as $t) {
    assert_iter_parity(NumPower::array([[1, 2], [3, 4], [5, 6]], $t), "2D-$t");
}

/* 3-D and 4-D: foreach yields sub-tensors. */
assert_iter_parity(NumPower::zeros([5, 2, 3]), "3D-5x2x3");
assert_iter_parity(NumPower::zeros([8, 2, 3, 4]), "4D-8x2x3x4");

/* Degenerate ranks. */
assert_iter_parity(NumPower::zeros([1, 1]), "1x1");
assert_iter_parity(NumPower::zeros([1, 99]), "1xN");

echo "done\n";
?>
--EXPECT--
OK 1D-float4 7
OK 1D-float8 7
OK 1D-float16 7
OK 1D-float32 7
OK 1D-float64 7
OK 1D-float128 7
OK 1D-int8 7
OK 1D-uint8 7
OK 1D-int16 7
OK 1D-uint16 7
OK 1D-int32 7
OK 1D-uint32 7
OK 1D-int64 7
OK 1D-uint64 7
OK 2D-float4 3
OK 2D-float8 3
OK 2D-float16 3
OK 2D-float32 3
OK 2D-float64 3
OK 2D-float128 3
OK 2D-int8 3
OK 2D-uint8 3
OK 2D-int16 3
OK 2D-uint16 3
OK 2D-int32 3
OK 2D-uint32 3
OK 2D-int64 3
OK 2D-uint64 3
OK 3D-5x2x3 5
OK 4D-8x2x3x4 8
OK 1x1 1
OK 1xN 1
done
