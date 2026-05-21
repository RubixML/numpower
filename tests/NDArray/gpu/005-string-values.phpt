--TEST--
NDArray::gpu()/cpu() preserves data created from PHP string values
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Confirm strings are an interchangeable input form across every dtype and
   survive a host↔device round-trip. For float128 / uint64 / int64 strings
   are the only lossless representation of out-of-range values. */
$cases = [
    'float4'   => [['0', '0.5', '1', '1.5'],                  [0, 0.5, 1, 1.5]],
    'float8'   => [['0', '0.5', '1', '1.5'],                  [0, 0.5, 1, 1.5]],
    'float16'  => [['0.25', '0.5', '0.75', '1'],              [0.25, 0.5, 0.75, 1]],
    'float32'  => [['0', '0.5', '1', '1.5'],                  [0, 0.5, 1, 1.5]],
    'float64'  => [['0', '0.5', '1', '1.5'],                  [0, 0.5, 1, 1.5]],
    'float128' => [['0', '0.5', '1', '1.5'],                  [0, 0.5, 1, 1.5]],
    'int8'     => [['-128', '0', '1', '127'],                 [-128, 0, 1, 127]],
    'uint8'    => [['0', '1', '128', '255'],                  [0, 1, 128, 255]],
    'int16'    => [['-32768', '0', '1', '32767'],             [-32768, 0, 1, 32767]],
    'uint16'   => [['0', '1', '32768', '65535'],              [0, 1, 32768, 65535]],
    'int32'    => [['-2147483648', '0', '1', '2147483647'],   [-2147483648, 0, 1, 2147483647]],
    'uint32'   => [['0', '1', '2147483648', '4294967295'],    [0, 1, 2147483648, 4294967295]],
    'int64'    => [['-1', '0', '1', '9223372036854775807'],   [-1, 0, 1, 9223372036854775807]],
    'uint64'   => [['0', '1', '2', '18446744073709551615'],   null], /* native PHP int can't hold the upper bound */
];

foreach ($cases as $dtype => $pair) {
    [$str_values, $num_values] = $pair;

    /* String input must survive a CPU -> GPU -> CPU round trip. */
    $a    = new NDArray($str_values, $dtype);
    $back = $a->gpu()->cpu();
    if ((string)$a !== (string)$back) {
        echo "$dtype: round-trip mismatch (string input)\n";
        continue;
    }

    /* Where comparable, string and numeric inputs must produce identical data. */
    if ($num_values !== null) {
        $b = new NDArray($num_values, $dtype);
        if ((string)$a !== (string)$b) {
            echo "$dtype: string vs numeric mismatch\n";
            continue;
        }
    }
    echo "$dtype: OK\n";
}
?>
--EXPECT--
float4: OK
float8: OK
float16: OK
float32: OK
float64: OK
float128: OK
int8: OK
uint8: OK
int16: OK
uint16: OK
int32: OK
uint32: OK
int64: OK
uint64: OK
