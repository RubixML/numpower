--TEST--
Unary string-scalar intake: no buffer-slot or VRAM leak across many iterations
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* The new string-scalar path allocates a transient 0-D NDArray via
   `ndarray_make_typed_scalar`, runs the unary op, and frees the
   transient back to the buffer pool. With NDARRAY_VCHECK=1 a surviving
   vmalloc'd buffer surfaces as a "VRAM MEMORY LEAK" line at RSHUTDOWN —
   the expected output therefore ends at "DONE" and contains NO leak
   diagnostic.

   Covers every op in MATHEMATICAL FUNCTIONS + EXPONENTIAL & LOGARITHMIC
   that flows through the shared `ndarray_run_simple_unary` plus `clip`
   which has its own helper. Each op runs over each of the three dtype
   inference branches (fp128 / int64 / uint64). */

$ops = ['abs','negative','positive','reciprocal','sign','sqrt','rsqrt',
        'square','sinc',
        'exp','exp2','expm1','log','log1p','log2','log10','logb'];

/* Three string literals, one per inference branch:
     - "1.5"  → fp128
     - "100"  → int64
     - "18446744073709551615" → uint64 */
$args = ['1.5', '100', '18446744073709551615'];

for ($iter = 0; $iter < 50; $iter++) {
    foreach ($args as $a) {
        foreach ($ops as $op) {
            try {
                $r = NumPower::$op($a);
                unset($r);
            } catch (\Throwable $t) {
                /* Skip log of negative / domain errors silently; the test
                   is about leaks, not return values. */
            }
        }
        try {
            $r = NumPower::clip($a, '0', '1');
            unset($r);
        } catch (\Throwable $t) { /* ignore */ }
    }
}

echo "DONE\n";
?>
--EXPECT--
DONE
