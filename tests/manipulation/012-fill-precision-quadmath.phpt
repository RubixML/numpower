--TEST--
NDArray::fill() preserves full fp128 precision (quadmath builds only)
--SKIPIF--
<?php
/* Probe: feed a string that fits inside fp128 (113 bits ~= 34 decimal) but
   not inside DD (~32 decimal). True quadmath round-trips it; DD
   canonicalizes it to a shorter representation. */
$probe = (string)(new NDArray(['9.999999999999999999999999999999999e-10'], 'float128'));
if (strpos($probe, '9.9999999999999999999999999999999') === false) {
    die('skip: float128 storage uses double-double emulation (no libquadmath)');
}
?>
--FILE--
<?php
/* fill() goes through the same string->fp128 path as the constructor, so on
   quadmath builds it must retain the full 34-digit precision the dtype can
   represent. The DD-safe subset of this check lives in 007-fill-precision.phpt. */
$a = new NDArray(['0','0','0'], 'float128');
$a->fill('3.14159265358979323846264338327950288');
foreach ($a->toArray() as $i => $v) {
    echo "fp128[$i]=$v\n";
}
?>
--EXPECT--
fp128[0]=3.141592653589793238462643383279503
fp128[1]=3.141592653589793238462643383279503
fp128[2]=3.141592653589793238462643383279503
