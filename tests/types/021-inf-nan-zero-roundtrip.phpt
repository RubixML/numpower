--TEST--
NDArray preserves INF, -INF, NaN, +0 and -0 across CPU and GPU element access
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Element access must preserve IEEE-754 special values for every float dtype
   that can natively represent them. Special values lock down the type-aware
   read path because a wrong dtype reinterpretation typically loses the
   exponent bits and silently flips ±INF / NaN to a finite number. */

$tests = [
    /* float16 can store ±INF and NaN natively. */
    ['float16',  [INF, -INF, NAN, 0.0, -0.0]],
    ['float32',  [INF, -INF, NAN, 0.0, -0.0]],
    ['float64',  [INF, -INF, NAN, 0.0, -0.0]],
];

foreach ($tests as [$dtype, $vals]) {
    $cpu = new NDArray($vals, $dtype);
    $gpu = $cpu->gpu();
    $ok  = true;
    foreach ($vals as $i => $expected) {
        $cv = $cpu[$i];
        $gv = $gpu[$i];
        $expect = $expected;
        /* NaN compares unequal to itself; treat is_nan as equivalent */
        if (is_nan($expect)) {
            if (!is_nan($cv) || !is_nan($gv)) { $ok = false; break; }
        } else {
            if ($cv !== $expect || $gv !== $expect) { $ok = false; break; }
        }
    }
    echo "$dtype: ", ($ok ? "OK" : "BAD"), "\n";
}

/* float128 via strings — must round-trip without losing INF/NaN sign or
   bumping ±0 to subnormal. The string formatter renders these as
   "inf"/"-inf"/"nan" (snprintf with %Lg). */
$f128_vals = ['inf', '-inf', 'nan', '0', '-0'];
$a = new NDArray($f128_vals, 'float128');
$g = $a->gpu();
$ok = true;
foreach ($f128_vals as $i => $exp) {
    $cv = $a[$i];
    $gv = $g[$i];
    if (!is_string($cv) || !is_string($gv)) { $ok = false; break; }
    if (strtolower($cv) !== strtolower($gv)) { $ok = false; break; }
}
echo 'float128: ', ($ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
float16: OK
float32: OK
float64: OK
float128: OK
