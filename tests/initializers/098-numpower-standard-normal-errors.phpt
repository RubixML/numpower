--TEST--
NumPower::standardNormal() raises clear errors on invalid dtype, device, or shape
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine in a sane state. standardNormal has fewer args than normal()
   (no loc/scale), so the surface is narrower — we still cover dtype,
   device, and shape rejection. */

try {
    NumPower::standardNormal([4], 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::standardNormal([4], 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* Negative shape entries are rejected by ndarray_parse_typed_shape. */
try {
    NumPower::standardNormal([-1], 'float32');
    echo "BAD: no throw on negative shape\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'negative dimensions')
        ? "neg_shape: OK\n"
        : "neg_shape: BAD ({$e->getMessage()})\n";
}

/* Negative shape entry combined with a valid device — same rejection
   path, but exercises that the device arg is parsed before the shape
   walk so no GPU memory is allocated before the throw. */
try {
    NumPower::standardNormal([2, -1], 'float32', 0);
    echo "BAD: no throw on mixed shape\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'negative dimensions')
        ? "neg_shape_mixed: OK\n"
        : "neg_shape_mixed: BAD ({$e->getMessage()})\n";
}

/* dtype passed as an array — Z_PARAM_STRING rejects with TypeError
   (array → string is not a valid PHP coercion). */
try {
    NumPower::standardNormal([4], ['float32']);
    echo "BAD: no throw on array dtype\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_dtype: OK\n"
        : "array_dtype: BAD ({$e->getMessage()})\n";
}

/* device passed as an array — Z_PARAM_LONG rejects with TypeError. */
try {
    NumPower::standardNormal([4], 'float32', ['gpu']);
    echo "BAD: no throw on array device\n";
} catch (\TypeError $e) {
    echo str_contains($e->getMessage(), 'must be')
        ? "array_device: OK\n"
        : "array_device: BAD ({$e->getMessage()})\n";
}

/* After every failure the engine still produces a correct standard
   normal — no internal state was corrupted. */
$ok = NumPower::standardNormal([4], 'float32');
echo 'recovered: ', ($ok->shape() === [4] ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
neg_shape: OK
neg_shape_mixed: OK
array_dtype: OK
array_device: OK
recovered: OK
