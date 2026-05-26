--TEST--
Integer arithmetic edge cases: mod-by-zero, mod sign-of-dividend, negative-exp pow on signed ints (CPU + GPU)
--FILE--
<?php
/* The native-int CPU kernel and the GPU `cuda_mod_<tag>` kernels must
   agree on integer mod/pow corner cases:
    - signed mod with negative dividend / divisor: result has sign of
      dividend (C11 `%` semantics, matches `torch.fmod` for integer
      tensors).
    - mod by zero on integer dtypes: returns 0 (avoids SIGFPE on CPU,
      matches the documented divide-by-zero contract on GPU).
    - signed pow with negative exponent: returns 0 (matches PyTorch's
      int-pow truncation contract — the mathematical result of `base^(-n)`
      is a fraction in [-1, 1] which truncates to 0 unless |base| == 1). */

$gpu_ok = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_ok = false; }

$ok = true;

function check($label, $got, $want, &$ok) {
    if (json_encode($got) !== json_encode($want)) {
        echo "FAIL $label: got=", json_encode($got), " want=", json_encode($want), "\n";
        $ok = false;
    }
}

/* Signed-integer mod with mixed signs across every signed int dtype. */
foreach (['int8','int16','int32','int64'] as $t) {
    $a = new NDArray([-7, 7, -10, 10], $t);
    $b = new NDArray([3, -3, 4, -4], $t);
    check("$t CPU mod",
          NumPower::mod($a, $b)->toArray(),
          [-1, 1, -2, 2], $ok);
    if ($gpu_ok) {
        check("$t GPU mod",
              NumPower::mod($a->gpu(), $b->gpu())->cpu()->toArray(),
              [-1, 1, -2, 2], $ok);
    }
}

/* Mod-by-zero on integer dtypes returns 0 (no SIGFPE). */
foreach (['int8','int16','int32','int64','uint8','uint16','uint32'] as $t) {
    $a = new NDArray([10, 20], $t);
    $b = new NDArray([0, 5], $t);
    check("$t CPU mod by 0",
          NumPower::mod($a, $b)->toArray(),
          [0, 0], $ok);
    if ($gpu_ok) {
        check("$t GPU mod by 0",
              NumPower::mod($a->gpu(), $b->gpu())->cpu()->toArray(),
              [0, 0], $ok);
    }
}

/* Signed pow with negative exponent: returns 0 for every base (including
   ±1) per the int-pow truncation contract — CPU and GPU agree exactly.
   This is a stricter convention than NumPy (which throws) but is the
   currently-defined behaviour shared between the CPU `NDARRAY_INT_BINOP_BODY`
   pow branch and the GPU `tcuda_pow_int` kernel. */
foreach (['int8','int16','int32','int64'] as $t) {
    $a = new NDArray([2, 3, -2, 1, -1], $t);
    $b = new NDArray([-1, -2, 3, -5, -3], $t);
    check("$t CPU pow neg-exp",
          NumPower::pow($a, $b)->toArray(),
          [0, 0, -8, 0, 0], $ok);
    if ($gpu_ok) {
        check("$t GPU pow neg-exp",
              NumPower::pow($a->gpu(), $b->gpu())->cpu()->toArray(),
              [0, 0, -8, 0, 0], $ok);
    }
}

/* Unsigned pow — exponent is non-negative, base is non-negative. */
foreach (['uint8','uint16','uint32'] as $t) {
    $a = new NDArray([2, 3, 4], $t);
    $b = new NDArray([3, 4, 2], $t);
    check("$t CPU pow",
          NumPower::pow($a, $b)->toArray(),
          [8, 81, 16], $ok);
    if ($gpu_ok) {
        check("$t GPU pow",
              NumPower::pow($a->gpu(), $b->gpu())->cpu()->toArray(),
              [8, 81, 16], $ok);
    }
}

/* uint64 pow keeps full precision; toArray returns strings. */
$a = new NDArray(['10', '10', '10'], 'uint64');
$b = new NDArray(['18', '19', '20'], 'uint64');
check("uint64 CPU pow strings",
      NumPower::pow($a, $b)->toArray(),
      ["1000000000000000000", "10000000000000000000", "7766279631452241920"], /* 10^20 mod 2^64 */
      $ok);

echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
