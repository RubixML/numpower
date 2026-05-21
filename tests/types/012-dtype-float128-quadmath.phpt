--TEST--
NDArray float128 dtype: full 34-digit precision (quadmath builds only)
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
/* These inputs exercise the extra precision native __float128 + libquadmath
   delivers beyond DD's ~32 decimal digits. */
$a = new NDArray(["3.14159265358979323846264338327950288"], "float128");
echo $a . "\n";
$b = new NDArray(["9.999999999999999999999999999999999e-10"], "float128");
echo $b . "\n";
?>
--EXPECT--
[3.141592653589793238462643383279503]
[9.999999999999999999999999999999999e-10]
