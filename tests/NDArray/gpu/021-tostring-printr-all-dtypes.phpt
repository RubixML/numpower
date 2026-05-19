--TEST--
__toString and print_r on GPU NDArrays produce the same output as CPU for every dtype
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* `print_r` is hijacked by the extension to route NDArray objects through
   NDArray_Print (same path as __toString). For GPU arrays the printer used
   to do a raw cudaMemcpy and then reinterpret the bytes as native dtype.
   This was wrong for float128, whose GPU layout is double-double (hi, lo) —
   the host stringifier read those 16 bytes as a native __float128 and
   produced garbage like `2.98e-4947`. Lock in that GPU print output matches
   CPU print output for every dtype, and that the high-precision fp128 value
   from the original bug report is rendered correctly. */

$cases = [
    'float4'   => ['0', '0.5', '1', '1.5'],
    'float8'   => ['0', '0.5', '1', '1.5'],
    'float16'  => ['0.25', '0.5', '0.75', '1'],
    'float32'  => ['0', '0.5', '1', '1.5'],
    'float64'  => ['0', '0.5', '1', '1.5'],
    /* fp128 GPU stores dd(hi, lo); restrict to fp64-exact values so the
       byte-for-byte CPU/GPU print comparison holds. */
    'float128' => ['0', '0.5', '1', '1.5'],
    'int8'     => ['-128', '-1', '0', '127'],
    'uint8'    => ['0', '1', '128', '255'],
    'int16'    => ['-32768', '-1', '0', '32767'],
    'uint16'   => ['0', '1', '32768', '65535'],
    'int32'    => ['-2147483648', '-1', '0', '2147483647'],
    'uint32'   => ['0', '1', '2147483648', '4294967295'],
    'int64'    => ['-9223372036854775808', '-1', '0', '9223372036854775807'],
    'uint64'   => ['0', '1', '9223372036854775808', '18446744073709551615'],
];

foreach ($cases as $dtype => $values) {
    $cpu = new NDArray($values, $dtype);
    $gpu = $cpu->gpu();

    $cpu_to_str = (string)$cpu;
    $gpu_to_str = (string)$gpu;
    $cpu_pr     = print_r($cpu, true);
    $gpu_pr     = print_r($gpu, true);

    $ok_to_str = ($cpu_to_str === $gpu_to_str);
    $ok_pr     = ($cpu_pr === $gpu_pr);
    /* Both entry points share the same code path — they must agree too. */
    $ok_match  = ($gpu_to_str === $gpu_pr);

    echo "$dtype: ",
         ($ok_to_str ? '__toString=OK' : '__toString=BAD'), ' ',
         ($ok_pr     ? 'print_r=OK'    : 'print_r=BAD'), ' ',
         ($ok_match  ? 'same=OK'       : 'same=BAD'), "\n";
}

/* Original bug report: high-precision fp128 string survives GPU print. */
$hp = '0.012345678909876543210123456782012345';
$arr = NumPower::array(['1', $hp], 'float128')->gpu();
$out = (string)$arr;
/* libquadmath stringifies fp128 to 34 significant digits, so the printed
   form is a 34-sig-digit prefix of the input; just check the leading
   digits to avoid coupling the test to the exact rounding. */
$leading = '0.0123456789098765432101234567820123';
if (str_contains($out, $leading) && str_contains($out, '1,')) {
    echo "fp128-hp: OK\n";
} else {
    echo "fp128-hp: BAD (got $out)\n";
}
?>
--EXPECT--
float4: __toString=OK print_r=OK same=OK
float8: __toString=OK print_r=OK same=OK
float16: __toString=OK print_r=OK same=OK
float32: __toString=OK print_r=OK same=OK
float64: __toString=OK print_r=OK same=OK
float128: __toString=OK print_r=OK same=OK
int8: __toString=OK print_r=OK same=OK
uint8: __toString=OK print_r=OK same=OK
int16: __toString=OK print_r=OK same=OK
uint16: __toString=OK print_r=OK same=OK
int32: __toString=OK print_r=OK same=OK
uint32: __toString=OK print_r=OK same=OK
int64: __toString=OK print_r=OK same=OK
uint64: __toString=OK print_r=OK same=OK
fp128-hp: OK
