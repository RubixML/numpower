--TEST--
clone $ndarray preserves do_operation handler so arithmetic still works
--FILE--
<?php
/* The default zend_objects_clone_obj path could install
   &std_object_handlers on the clone instead of the custom
   &ndarray_object_handlers, dropping the do_operation slot. After that,
   $clone + N would throw "Unsupported operand types: NDArray + int"
   even though the original $a + N worked. The custom ndarray_clone_obj
   handler now allocates the clone via the same code path as
   ndarray_create_object so the handler table is always wired up. */

/* 1-D float32 */
$a = new NDArray([1.0, 2.0, 3.0], 'float32');
$b = clone $a;
$r = $b + 2;
echo "1-D + int: ", ($r instanceof NDArray && $r->toArray() === [3.0, 4.0, 5.0]) ? "ok\n" : "FAIL\n";

/* 0-D float64 — clone + scalar produces a 0-D result (collapsed to a
   PHP scalar by NumPower's ndarray_init_new_object — that's expected
   for primitive-returning ops). */
$s = new NDArray(7, 'float64');
$sc = clone $s;
$rs = $sc + 3;
echo "0-D + int: ", (is_int($rs) || is_float($rs)) && (float) $rs === 10.0 ? "ok\n" : "FAIL\n";

/* String concat (do_operation must return FAILURE for non-arith opcodes
   so PHP falls back to __toString). */
$c = clone $a;
$concat = "got: " . $c;
echo "to-string: ", strlen($concat) > 0 ? "ok\n" : "FAIL\n";

/* Other arithmetic ops also work on clones. */
$d = clone new NDArray([10.0, 20.0], 'float32');
$ok = (($d - 1)->toArray() === [9.0, 19.0])
   && (($d * 2)->toArray() === [20.0, 40.0])
   && (($d / 2)->toArray() === [5.0, 10.0])
   && (($d ** 2)->toArray() === [100.0, 400.0]);
echo "ops: ", $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
1-D + int: ok
0-D + int: ok
to-string: ok
ops: ok
