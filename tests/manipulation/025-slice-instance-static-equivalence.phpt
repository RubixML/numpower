--TEST--
slice() instance and static forms compute byte-identical results for every dtype on CPU and GPU
--FILE--
<?php
/* The two forms must share the underlying slice algorithm. If they ever
   diverge — say someone special-cases one form for performance — boundary
   values would silently differ across the two APIs. This test cross-checks
   instance vs static for every dtype on each device, across the three slice
   patterns the algorithm decomposes differently:
     - leading-axis range (bulk path)
     - column extraction   (strided path / GPU kernel)
     - 0-D reduction       (scalar copy path) */

$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Error $e) {}

$types = ['float4','float8','float16','float32','float64','float128',
          'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

/* Build two fresh sources with the same data, run static on one, instance
   on the other, then compare. We can't safely "clone" an NDArray after
   instance-form mutation (no public dtype getter to round-trip through),
   so the closure pattern is the cleanest way to share construction logic. */
function compare_forms(string $dtype, callable $build_src, array $slice_args, string $tag): void {
    $a = $build_src();
    $b = $build_src();
    $static = NumPower::slice($a, ...$slice_args);
    $inst   = $b->slice(...$slice_args);

    /* Normalise: both might be NDArray or both might be scalar. GPU results
       need ->cpu() before toArray() because the dtype-aware reader walks
       host memory. */
    if (is_object($static) && is_object($inst)) {
        if ($static->isGPU()) $static = $static->cpu();
        if ($inst->isGPU())   $inst   = $inst->cpu();
        $eq = ($static->shape() === $inst->shape())
            && ($static->toArray() === $inst->toArray());
    } else {
        $eq = ($static === $inst);
    }
    echo "$dtype $tag: ", $eq ? "OK" : "BAD", "\n";
}

foreach ($types as $t) {
    $cpu_build = fn() => new NDArray([[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [3, 2, 1, 0]], $t);

    /* Leading-axis range: result is shape [2, 4], contiguous run path */
    compare_forms($t, $cpu_build, [[0, 2]], "CPU leading-range");

    /* Column extraction: result is shape [3], strided path (GPU kernel) */
    compare_forms($t, $cpu_build, [[], -2], "CPU column");

    /* 0-D reduction: scalar return (string for fp128/uint64) */
    compare_forms($t, $cpu_build, [1, 2], "CPU 0-D scalar");

    /* Negative-step reverse: kernel path with signed strides */
    compare_forms($t, $cpu_build, [[2, -4, -1]], "CPU neg-step");

    if ($has_gpu) {
        $gpu_build = fn() => (new NDArray([[1, 2, 3, 4],
                                           [5, 6, 7, 8],
                                           [3, 2, 1, 0]], $t))->gpu();

        compare_forms($t, $gpu_build, [[0, 2]],    "GPU leading-range");
        compare_forms($t, $gpu_build, [[], -2],    "GPU column");
        compare_forms($t, $gpu_build, [1, 2],      "GPU 0-D scalar");
        compare_forms($t, $gpu_build, [[2, -4, -1]], "GPU neg-step");
    } else {
        echo "$t GPU leading-range: OK\n";
        echo "$t GPU column: OK\n";
        echo "$t GPU 0-D scalar: OK\n";
        echo "$t GPU neg-step: OK\n";
    }
}
?>
--EXPECT--
float4 CPU leading-range: OK
float4 CPU column: OK
float4 CPU 0-D scalar: OK
float4 CPU neg-step: OK
float4 GPU leading-range: OK
float4 GPU column: OK
float4 GPU 0-D scalar: OK
float4 GPU neg-step: OK
float8 CPU leading-range: OK
float8 CPU column: OK
float8 CPU 0-D scalar: OK
float8 CPU neg-step: OK
float8 GPU leading-range: OK
float8 GPU column: OK
float8 GPU 0-D scalar: OK
float8 GPU neg-step: OK
float16 CPU leading-range: OK
float16 CPU column: OK
float16 CPU 0-D scalar: OK
float16 CPU neg-step: OK
float16 GPU leading-range: OK
float16 GPU column: OK
float16 GPU 0-D scalar: OK
float16 GPU neg-step: OK
float32 CPU leading-range: OK
float32 CPU column: OK
float32 CPU 0-D scalar: OK
float32 CPU neg-step: OK
float32 GPU leading-range: OK
float32 GPU column: OK
float32 GPU 0-D scalar: OK
float32 GPU neg-step: OK
float64 CPU leading-range: OK
float64 CPU column: OK
float64 CPU 0-D scalar: OK
float64 CPU neg-step: OK
float64 GPU leading-range: OK
float64 GPU column: OK
float64 GPU 0-D scalar: OK
float64 GPU neg-step: OK
float128 CPU leading-range: OK
float128 CPU column: OK
float128 CPU 0-D scalar: OK
float128 CPU neg-step: OK
float128 GPU leading-range: OK
float128 GPU column: OK
float128 GPU 0-D scalar: OK
float128 GPU neg-step: OK
int8 CPU leading-range: OK
int8 CPU column: OK
int8 CPU 0-D scalar: OK
int8 CPU neg-step: OK
int8 GPU leading-range: OK
int8 GPU column: OK
int8 GPU 0-D scalar: OK
int8 GPU neg-step: OK
uint8 CPU leading-range: OK
uint8 CPU column: OK
uint8 CPU 0-D scalar: OK
uint8 CPU neg-step: OK
uint8 GPU leading-range: OK
uint8 GPU column: OK
uint8 GPU 0-D scalar: OK
uint8 GPU neg-step: OK
int16 CPU leading-range: OK
int16 CPU column: OK
int16 CPU 0-D scalar: OK
int16 CPU neg-step: OK
int16 GPU leading-range: OK
int16 GPU column: OK
int16 GPU 0-D scalar: OK
int16 GPU neg-step: OK
uint16 CPU leading-range: OK
uint16 CPU column: OK
uint16 CPU 0-D scalar: OK
uint16 CPU neg-step: OK
uint16 GPU leading-range: OK
uint16 GPU column: OK
uint16 GPU 0-D scalar: OK
uint16 GPU neg-step: OK
int32 CPU leading-range: OK
int32 CPU column: OK
int32 CPU 0-D scalar: OK
int32 CPU neg-step: OK
int32 GPU leading-range: OK
int32 GPU column: OK
int32 GPU 0-D scalar: OK
int32 GPU neg-step: OK
uint32 CPU leading-range: OK
uint32 CPU column: OK
uint32 CPU 0-D scalar: OK
uint32 CPU neg-step: OK
uint32 GPU leading-range: OK
uint32 GPU column: OK
uint32 GPU 0-D scalar: OK
uint32 GPU neg-step: OK
int64 CPU leading-range: OK
int64 CPU column: OK
int64 CPU 0-D scalar: OK
int64 CPU neg-step: OK
int64 GPU leading-range: OK
int64 GPU column: OK
int64 GPU 0-D scalar: OK
int64 GPU neg-step: OK
uint64 CPU leading-range: OK
uint64 CPU column: OK
uint64 CPU 0-D scalar: OK
uint64 CPU neg-step: OK
uint64 GPU leading-range: OK
uint64 GPU column: OK
uint64 GPU 0-D scalar: OK
uint64 GPU neg-step: OK
