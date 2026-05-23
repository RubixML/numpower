--TEST--
clone $ndarray produces a deep copy — modifying the clone leaves the source untouched
--FILE--
<?php
/* Without an explicit clone handler, both the original and the clone
   would have the same private `id` (UUID) and therefore share the same
   underlying buffer slot. The custom ndarray_clone_obj handler calls
   NDArray_Copy to give the clone its own slot, so writes to the clone
   stay local. */

/* 1-D: in-place mutation via offsetSet should NOT bleed back to source. */
$a = new NDArray([1.0, 2.0, 3.0], 'float32');
$b = clone $a;
$b[0] = 99;
$ok1 = $a->toArray() === [1.0, 2.0, 3.0] && $b->toArray() === [99.0, 2.0, 3.0];
echo "1-D mutation: ", $ok1 ? "ok\n" : "FAIL a=" . json_encode($a->toArray())
                              . " b=" . json_encode($b->toArray()) . "\n";

/* 2-D: same check across rows. */
$m  = new NDArray([[1.0, 2.0], [3.0, 4.0]], 'float32');
$mc = clone $m;
$mc[0] = new NDArray([7.0, 8.0], 'float32');
$ok2 = $m->toArray() === [[1.0, 2.0], [3.0, 4.0]]
    && $mc->toArray() === [[7.0, 8.0], [3.0, 4.0]];
echo "2-D mutation: ", $ok2 ? "ok\n" : "FAIL\n";

/* Type / shape preserved across clone. */
$d  = new NDArray([[1, 2, 3]], 'int32');
$dc = clone $d;
$ok3 = $dc->shape() === [1, 3] && $dc->toArray() === [[1, 2, 3]];
echo "shape/dtype: ", $ok3 ? "ok\n" : "FAIL\n";
?>
--EXPECT--
1-D mutation: ok
2-D mutation: ok
shape/dtype: ok
