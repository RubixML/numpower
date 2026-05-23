--TEST--
NumPower::setDevice() accepts PHP int-coercible values (loose typing)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Z_PARAM_LONG accepts standard PHP coercion in non-strict mode: numeric
   strings, finite floats, and booleans coerce to int. The signature in
   stub.php declares `int $deviceId`, so a strict-typed caller still
   gets a TypeError; loose-typed code (the default) coerces. We assert
   the loose-mode path here so future refactors don't accidentally
   tighten the arginfo. */
/* PHP coerces these all to int 0 in non-strict mode. Boolean `true`
   coerces to 1, which is device-count dependent (succeeds on 2+ GPUs,
   throws on 1 GPU) — that path is covered by 023-setdevice-bool-false. */
$cases = [
    'int 0'        => 0,
    'numeric "0"'  => "0",
    'float 0.0'    => 0.0,
    'false'        => false,
];
foreach ($cases as $label => $val) {
    try {
        NumPower::setDevice($val);
        echo "$label: ok\n";
    } catch (\Throwable $e) {
        echo "$label: FAIL ", get_class($e), " ", $e->getMessage(), "\n";
    }
}
?>
--EXPECT--
int 0: ok
numeric "0": ok
float 0.0: ok
false: ok
