--TEST--
NumPower::syncDevice() and NumPower::dumpDevices() reject extra arguments
--FILE--
<?php
/* All three device methods declared 0-arg in the C arginfo. Both
   syncDevice and dumpDevices added ZEND_PARSE_PARAMETERS_START(0, 0)
   to enforce that at call time; setDevice's existing 1-arg parse stays
   in place. Passing extra positional args must surface as either
   ArgumentCountError or plain Error (PHP version-dependent). */
$cases = [
    'syncDevice(1)'     => function () { NumPower::syncDevice(1); },
    'syncDevice(1,2)'   => function () { NumPower::syncDevice(1, 2); },
    'dumpDevices(1)'    => function () { NumPower::dumpDevices(1); },
    'setDevice()'       => function () { NumPower::setDevice(); },
    'setDevice(0, 1)'   => function () { NumPower::setDevice(0, 1); },
];
foreach ($cases as $label => $fn) {
    try {
        $fn();
        echo "$label: FAIL no throw\n";
    } catch (\ArgumentCountError $e) {
        echo "$label: ok\n";
    } catch (\Error $e) {
        echo "$label: ok\n";
    }
}
?>
--EXPECT--
syncDevice(1): ok
syncDevice(1,2): ok
dumpDevices(1): ok
setDevice(): ok
setDevice(0, 1): ok
