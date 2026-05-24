--TEST--
Passing an existing NDArray as $shape to zeros / ones / full keeps the source alive
--FILE--
<?php
/* Pre-existing UAF: `ndarray_parse_typed_shape` called `NDArray_FREE(nda)`
 * unconditionally after `ZVAL_TO_NDARRAY`. For an existing NDArray
 * source the call returns the buffer entry without bumping the
 * refcount; the bare `NDArray_FREE` then dropped the user's array to
 * refcount 0 and freed it, leaving the PHP variable pointing at a
 * dangling buffer slot. Symptom: a subsequent `echo $shape` printed
 * arbitrary recycled memory (often the just-built result matrix
 * because the Zend allocator immediately reused the slot).
 *
 * The fix replaces the bare NDArray_FREE with `CHECK_INPUT_AND_FREE`,
 * which only releases the temporary NDArray when the source zval was a
 * scalar / array. */

/* Construct a shape NDArray, capture its string representation, then
   pass it to each of the three affected factories and verify the
   shape variable still produces the same string afterwards. */
$shape  = NumPower::array([4, 4]);
$before = (string)$shape;

$z = NumPower::zeros($shape);
echo 'zeros: shape_intact=',
     ((string)$shape === $before ? 'OK' : 'BAD'), "\n";
echo 'zeros result shape: ', json_encode($z->shape()), "\n";

$o = NumPower::ones($shape);
echo 'ones: shape_intact=',
     ((string)$shape === $before ? 'OK' : 'BAD'), "\n";

$f = NumPower::full($shape, 7, 'int32');
echo 'full: shape_intact=',
     ((string)$shape === $before ? 'OK' : 'BAD'), "\n";

/* Same NDArray reused across many calls — refcount must remain stable. */
for ($i = 0; $i < 20; $i++) {
    NumPower::zeros($shape);
}
echo 'after_20_zeros: shape_intact=',
     ((string)$shape === $before ? 'OK' : 'BAD'), "\n";

/* The shape NDArray is still usable for ordinary ops. */
$twice = NumPower::add($shape, $shape);
echo 'shape_still_arithmetic_capable: ', (string)$twice, "\n";

/* PHP array source still works (no temp leak). */
$z = NumPower::zeros([4, 4]);
echo 'php_array_source_ok: ',
     ($z->shape() === [4, 4] ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
zeros: shape_intact=OK
zeros result shape: [4,4]
ones: shape_intact=OK
full: shape_intact=OK
after_20_zeros: shape_intact=OK
shape_still_arithmetic_capable: [8, 8]
php_array_source_ok: OK
