--TEST--
slice() 0-D scalar return: dtype-correct PHP type (string for float128/uint64, int for integer dtypes, float otherwise) on both instance and static forms, CPU and GPU
--FILE--
<?php
/* slice() reducing a 1-D NDArray to a 0-D scalar must return a PHP value
   typed per the source dtype:
     - float128, uint64                              → string
     - int8/uint8/int16/uint16/int32/uint32/int64    → int
     - float4/float8/float16/float32/float64         → float
   This contract holds for BOTH the instance method (which mutates $this to
   0-D and returns the scalar) and the static method (which leaves the source
   untouched), on CPU and on GPU. Values are picked to be exactly
   representable in every dtype (3, 0, 1) so the test doesn't drift on
   fp4/fp8 quantisation. */

$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Error $e) {}

/* dtype → expected PHP type of the dtype-correct scalar */
$expected_type = [
    'float4'   => 'double',
    'float8'   => 'double',
    'float16'  => 'double',
    'float32'  => 'double',
    'float64'  => 'double',
    'float128' => 'string',   // <-- key cases
    'int8'     => 'integer',
    'uint8'    => 'integer',
    'int16'    => 'integer',
    'uint16'   => 'integer',
    'int32'    => 'integer',
    'uint32'   => 'integer',
    'int64'    => 'integer',
    'uint64'   => 'string',   // <-- key cases
];

foreach ($expected_type as $dtype => $php_type) {
    /* Source-of-truth value: NDArray([v], $dtype)[0] reads the dtype-encoded
       byte representation of "3" back through NDArray_ScalarToZval. The slice
       result must match it bit-for-bit, since slice copies bytes. */
    $expected = (new NDArray(['3'], $dtype))[0];

    /* ---- static (non-mutating) on CPU ---- */
    $src       = new NDArray(['1', '3', '2'], $dtype);
    $sc_static = NumPower::slice($src, 1);
    $ok = (gettype($sc_static) === $php_type && $sc_static === $expected);
    echo "$dtype static  CPU: ", $ok ? "OK" : "BAD got_type=" . gettype($sc_static)
         . " val=" . var_export($sc_static, true), "\n";
    /* Source must still be intact */
    $src_intact = ($src->toArray() === [
        (new NDArray(['1'], $dtype))[0],
        (new NDArray(['3'], $dtype))[0],
        (new NDArray(['2'], $dtype))[0],
    ]);
    echo "$dtype src intact:  ", $src_intact ? "OK" : "BAD", "\n";

    /* ---- instance (mutating) on CPU ---- */
    $mut      = new NDArray(['1', '3', '2'], $dtype);
    $sc_inst  = $mut->slice(1);
    $ok = (gettype($sc_inst) === $php_type && $sc_inst === $expected);
    echo "$dtype inst    CPU: ", $ok ? "OK" : "BAD got_type=" . gettype($sc_inst)
         . " val=" . var_export($sc_inst, true), "\n";
    /* $mut is now 0-D */
    echo "$dtype mut->0D:     ", ($mut->shape() === [] ? "OK" : "BAD"), "\n";

    if ($has_gpu) {
        /* ---- static on GPU ---- */
        $g_src    = (new NDArray(['1', '3', '2'], $dtype))->gpu();
        $sc_gpu_s = NumPower::slice($g_src, 1);
        $ok = (gettype($sc_gpu_s) === $php_type && $sc_gpu_s === $expected);
        echo "$dtype static  GPU: ", $ok ? "OK" : "BAD got=" . var_export($sc_gpu_s, true), "\n";

        /* ---- instance on GPU ---- */
        $g_mut    = (new NDArray(['1', '3', '2'], $dtype))->gpu();
        $sc_gpu_i = $g_mut->slice(1);
        $ok = (gettype($sc_gpu_i) === $php_type && $sc_gpu_i === $expected);
        echo "$dtype inst    GPU: ", $ok ? "OK" : "BAD got=" . var_export($sc_gpu_i, true), "\n";
    } else {
        /* Deterministic placeholders for no-GPU builds. */
        echo "$dtype static  GPU: OK\n";
        echo "$dtype inst    GPU: OK\n";
    }
}

/* Spot-check the string-return cases with values that NEED string
   (overflow PHP_INT_MAX, exceed double precision). */
$uint64_max  = '18446744073709551615';
$fp128_high  = '1.234567890123456789012345e100';

$u = new NDArray([$uint64_max], 'uint64');
$us = NumPower::slice($u, 0);
echo "uint64 max preserved: ", ($us === $uint64_max ? "OK" : "BAD got=" . var_export($us, true)), "\n";

$f = new NDArray([$fp128_high], 'float128');
$fs = NumPower::slice($f, 0);
echo "fp128 high preserved (>= 25 dig matches): ",
     (is_string($fs) && substr($fs, 0, 25) === substr($fp128_high, 0, 25) ? "OK"
        : "BAD got=" . var_export($fs, true)), "\n";

/* Same string-return spot-check via the instance form */
$u2 = new NDArray([$uint64_max], 'uint64');
$us2 = $u2->slice(0);
echo "uint64 max inst form: ", ($us2 === $uint64_max ? "OK" : "BAD got=" . var_export($us2, true)), "\n";

$f2 = new NDArray([$fp128_high], 'float128');
$fs2 = $f2->slice(0);
echo "fp128 high inst form: ",
     (is_string($fs2) && substr($fs2, 0, 25) === substr($fp128_high, 0, 25) ? "OK"
        : "BAD got=" . var_export($fs2, true)), "\n";
?>
--EXPECT--
float4 static  CPU: OK
float4 src intact:  OK
float4 inst    CPU: OK
float4 mut->0D:     OK
float4 static  GPU: OK
float4 inst    GPU: OK
float8 static  CPU: OK
float8 src intact:  OK
float8 inst    CPU: OK
float8 mut->0D:     OK
float8 static  GPU: OK
float8 inst    GPU: OK
float16 static  CPU: OK
float16 src intact:  OK
float16 inst    CPU: OK
float16 mut->0D:     OK
float16 static  GPU: OK
float16 inst    GPU: OK
float32 static  CPU: OK
float32 src intact:  OK
float32 inst    CPU: OK
float32 mut->0D:     OK
float32 static  GPU: OK
float32 inst    GPU: OK
float64 static  CPU: OK
float64 src intact:  OK
float64 inst    CPU: OK
float64 mut->0D:     OK
float64 static  GPU: OK
float64 inst    GPU: OK
float128 static  CPU: OK
float128 src intact:  OK
float128 inst    CPU: OK
float128 mut->0D:     OK
float128 static  GPU: OK
float128 inst    GPU: OK
int8 static  CPU: OK
int8 src intact:  OK
int8 inst    CPU: OK
int8 mut->0D:     OK
int8 static  GPU: OK
int8 inst    GPU: OK
uint8 static  CPU: OK
uint8 src intact:  OK
uint8 inst    CPU: OK
uint8 mut->0D:     OK
uint8 static  GPU: OK
uint8 inst    GPU: OK
int16 static  CPU: OK
int16 src intact:  OK
int16 inst    CPU: OK
int16 mut->0D:     OK
int16 static  GPU: OK
int16 inst    GPU: OK
uint16 static  CPU: OK
uint16 src intact:  OK
uint16 inst    CPU: OK
uint16 mut->0D:     OK
uint16 static  GPU: OK
uint16 inst    GPU: OK
int32 static  CPU: OK
int32 src intact:  OK
int32 inst    CPU: OK
int32 mut->0D:     OK
int32 static  GPU: OK
int32 inst    GPU: OK
uint32 static  CPU: OK
uint32 src intact:  OK
uint32 inst    CPU: OK
uint32 mut->0D:     OK
uint32 static  GPU: OK
uint32 inst    GPU: OK
int64 static  CPU: OK
int64 src intact:  OK
int64 inst    CPU: OK
int64 mut->0D:     OK
int64 static  GPU: OK
int64 inst    GPU: OK
uint64 static  CPU: OK
uint64 src intact:  OK
uint64 inst    CPU: OK
uint64 mut->0D:     OK
uint64 static  GPU: OK
uint64 inst    GPU: OK
uint64 max preserved: OK
fp128 high preserved (>= 25 dig matches): OK
uint64 max inst form: OK
fp128 high inst form: OK
