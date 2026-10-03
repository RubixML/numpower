--TEST--
NDArray::cast() / setDataType() reject invalid arguments and leave the array unchanged
--FILE--
<?php
$cases = [
    'no args'        => fn($a) => $a->cast(),
    'two args'       => fn($a) => $a->cast('int8', 'int8'),
    'array'          => fn($a) => $a->cast([]),
    'object'         => fn($a) => $a->cast(new stdClass()),
    'empty string'   => fn($a) => $a->cast(''),
    'unknown'        => fn($a) => $a->cast('float17'),
    'wrong case'     => fn($a) => $a->cast('Float32'),
    'trailing space' => fn($a) => $a->cast('float16 '),
    'NUL byte'       => fn($a) => $a->cast("float16\0junk"),
    'NUL byte (set)' => fn($a) => $a->setDataType("int8\0"),
    'numeric'        => fn($a) => $a->cast(123),
];

foreach ($cases as $label => $fn) {
    $a = new NDArray([1.5, 2.5], 'float32');
    try {
        $fn($a);
        $res = 'no exception';
    } catch (Throwable $e) {
        $res = get_class($e);
    }
    $intact = $a->dataType() === 'float32' && $a->toArray() === [1.5, 2.5];
    echo str_pad($label, 15), ' ', $res, ' ', $intact ? 'intact' : 'CHANGED', "\n";
}
?>
--EXPECT--
no args         ArgumentCountError intact
two args        ArgumentCountError intact
array           TypeError intact
object          TypeError intact
empty string    Error intact
unknown         Error intact
wrong case      Error intact
trailing space  Error intact
NUL byte        Error intact
NUL byte (set)  Error intact
numeric         Error intact
