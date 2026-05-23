--TEST--
NumPower::array accepts PHP arrays with sparse / non-contiguous keys (1-D, CPU)
--FILE--
<?php
/* Regression: NumPower::array($x, $dtype) used to return NULL whenever $x had
 * non-contiguous integer keys (e.g. after unset, or string keys, or a non-zero
 * starting key). Iteration must now follow PHP's insertion order, matching
 * foreach semantics. We use values 1..4 so the result round-trips losslessly
 * across every supported dtype, including float4 (E2M1, representable set
 * {0, 0.5, 1, 1.5, 2, 3, 4, 6}). */

$cases = [
    'unset_first'   => function () { $x = [9, 1, 2, 3, 4]; unset($x[0]); return $x; },
    'unset_middle'  => function () { $x = [1, 2, 9, 3, 4]; unset($x[2]); return $x; },
    'string_keys'   => function () { return ['a' => 1, 'b' => 2, 'c' => 3, 'd' => 4]; },
    'offset_start'  => function () { return [5 => 1, 6 => 2, 7 => 3, 8 => 4]; },
    'negative_keys' => function () { return [-2 => 1, -1 => 2, 0 => 3, 1 => 4]; },
];

$expected = [1, 2, 3, 4];

$dtypes = [
    'float4', 'float8', 'float16', 'float32', 'float64', 'float128',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
];

foreach ($dtypes as $dt) {
    foreach ($cases as $label => $maker) {
        $arr = NumPower::array($maker(), $dt);
        $shape = $arr->shape();
        $got = $arr->toArray();

        /* int/uint64 + float128 return strings via toArray(); rest return numbers. */
        $got_num = array_map(fn($v) => is_string($v) ? (int) $v : (int) $v, $got);

        $ok = ($shape === [4]) && ($got_num === $expected);
        echo $dt, ' ', $label, ': ', $ok ? 'OK' : 'FAIL', "\n";
        if (!$ok) {
            echo "  expected: ", json_encode($expected), "\n";
            echo "  got:      ", json_encode($got), "\n";
            echo "  shape:    ", json_encode($shape), "\n";
        }
    }
}

/* The user's exact failing snippet must now compute 1+2, 2+3, 3+4, 4+5. */
$x = [1, 2, 3, 4, 5];
unset($x[0]);
$a = NumPower::array([1, 2, 3, 4], 'float128');
$b = NumPower::array($x, 'float128');
echo "user_snippet: ", ($a + $b), "\n";
?>
--EXPECT--
float4 unset_first: OK
float4 unset_middle: OK
float4 string_keys: OK
float4 offset_start: OK
float4 negative_keys: OK
float8 unset_first: OK
float8 unset_middle: OK
float8 string_keys: OK
float8 offset_start: OK
float8 negative_keys: OK
float16 unset_first: OK
float16 unset_middle: OK
float16 string_keys: OK
float16 offset_start: OK
float16 negative_keys: OK
float32 unset_first: OK
float32 unset_middle: OK
float32 string_keys: OK
float32 offset_start: OK
float32 negative_keys: OK
float64 unset_first: OK
float64 unset_middle: OK
float64 string_keys: OK
float64 offset_start: OK
float64 negative_keys: OK
float128 unset_first: OK
float128 unset_middle: OK
float128 string_keys: OK
float128 offset_start: OK
float128 negative_keys: OK
int8 unset_first: OK
int8 unset_middle: OK
int8 string_keys: OK
int8 offset_start: OK
int8 negative_keys: OK
uint8 unset_first: OK
uint8 unset_middle: OK
uint8 string_keys: OK
uint8 offset_start: OK
uint8 negative_keys: OK
int16 unset_first: OK
int16 unset_middle: OK
int16 string_keys: OK
int16 offset_start: OK
int16 negative_keys: OK
uint16 unset_first: OK
uint16 unset_middle: OK
uint16 string_keys: OK
uint16 offset_start: OK
uint16 negative_keys: OK
int32 unset_first: OK
int32 unset_middle: OK
int32 string_keys: OK
int32 offset_start: OK
int32 negative_keys: OK
uint32 unset_first: OK
uint32 unset_middle: OK
uint32 string_keys: OK
uint32 offset_start: OK
uint32 negative_keys: OK
int64 unset_first: OK
int64 unset_middle: OK
int64 string_keys: OK
int64 offset_start: OK
int64 negative_keys: OK
uint64 unset_first: OK
uint64 unset_middle: OK
uint64 string_keys: OK
uint64 offset_start: OK
uint64 negative_keys: OK
user_snippet: [3, 5, 7, 9]
