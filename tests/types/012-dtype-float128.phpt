--TEST--
NDArray float128 dtype: creation from strings, high-precision display
--FILE--
<?php
/* Values are chosen to fit cleanly inside ~32 decimal digits — the precision
   tier guaranteed across every supported build:
     - Linux GCC + libquadmath: native __float128 (~34 decimal digits)
     - Apple clang / MSVC:      double-double emulation (~32 decimal digits)
   For inputs that exercise the *full* quadmath-only precision (>32 digits)
   see tests/types/012-dtype-float128-quadmath.phpt. */
$a = new NDArray(["1.23456789012345678901234"], "float128");
echo $a . "\n";
$b = new NDArray(["3.1415926535897932384626433"], "float128");
echo $b . "\n";
?>
--EXPECT--
[1.23456789012345678901234]
[3.1415926535897932384626433]
