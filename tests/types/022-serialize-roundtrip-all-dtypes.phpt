--TEST--
serialize() / unserialize() round-trip preserves dtype and values for every dtype
--FILE--
<?php
/* The earlier toArray() fix returned strings for float128 / uint64. Without
   embedding dtype metadata, those strings would have failed to unserialize
   (Create_NDArray_FromZval default-routes through float32 which doesn't
   accept string elements). __serialize/__unserialize now use a structured
   payload — this test locks down the round-trip across every dtype. */

$cases = [
    'float4'   => ['1.5', '-1.5', '2'],
    'float8'   => ['1.5', '-1.5', '2', '6'],
    'float16'  => ['1.5', '-1.5', '0.5', '-0.5'],
    'float32'  => [1.5, 2.5, -3.5],
    'float64'  => [1.5, 2.5, -3.5],
    'float128' => ['3.141592653589793238', '-1.5', '1e-9'],
    'int8'     => [-128, 0, 127],
    'uint8'    => [0, 128, 255],
    'int16'    => [-32768, 0, 32767],
    'uint16'   => [0, 32768, 65535],
    'int32'    => [-2147483648, 0, 2147483647],
    'uint32'   => [0, 2147483648, 4294967295],
    'int64'    => [PHP_INT_MIN, 0, PHP_INT_MAX],
    'uint64'   => ['0', '1', '18446744073709551615'],
];

foreach ($cases as $t => $vals) {
    $a = new NDArray($vals, $t);
    $b = unserialize(serialize($a));
    /* Value parity */
    $value_ok = ($a->toArray() === $b->toArray());
    /* dtype preserved — verify by checking PHP type of first element */
    $atype = gettype($a->toArray()[0]);
    $btype = gettype($b->toArray()[0]);
    $type_ok = ($atype === $btype);
    /* __toString equivalence — locks down the round-trip end-to-end */
    $str_ok = ((string)$a === (string)$b);
    echo "$t: ", ($value_ok ? "val=OK" : "val=BAD"), " ",
                  ($type_ok ? "type=OK" : "type=BAD"), " ",
                  ($str_ok ? "str=OK" : "str=BAD"), "\n";
}
?>
--EXPECT--
float4: val=OK type=OK str=OK
float8: val=OK type=OK str=OK
float16: val=OK type=OK str=OK
float32: val=OK type=OK str=OK
float64: val=OK type=OK str=OK
float128: val=OK type=OK str=OK
int8: val=OK type=OK str=OK
uint8: val=OK type=OK str=OK
int16: val=OK type=OK str=OK
uint16: val=OK type=OK str=OK
int32: val=OK type=OK str=OK
uint32: val=OK type=OK str=OK
int64: val=OK type=OK str=OK
uint64: val=OK type=OK str=OK
