--TEST--
NumPower::arange() raises clear errors on step==0, invalid dtype/device, or unacceptable args
--FILE--
<?php
/* Every validation branch must throw a catchable Error and leave the
   engine in a sane state. */

try {
    NumPower::arange(10, 0, 0);
    echo "BAD: no throw on step=0\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'step must be non-zero')
        ? "step_zero: OK\n"
        : "step_zero: BAD ({$e->getMessage()})\n";
}

try {
    NumPower::arange(NAN, 0, 1);
    echo "BAD: no throw on NaN stop\n";
} catch (\Error $e) {
    echo str_contains($e->getMessage(), 'step must be non-zero')
        ? "nan: OK\n"
        : "nan: BAD ({$e->getMessage()})\n";
}

try {
    NumPower::arange(5, 0, 1, 'banana');
    echo "BAD: no throw on bogus dtype\n";
} catch (\Error $e) {
    echo str_starts_with($e->getMessage(), "Invalid data type 'banana'")
        ? "dtype: OK\n"
        : "dtype: BAD ({$e->getMessage()})\n";
}

foreach ([2, -1, 99] as $bad) {
    try {
        NumPower::arange(5, 0, 1, 'float32', $bad);
        echo "BAD: no throw on device=$bad\n";
    } catch (\Error $e) {
        echo str_starts_with($e->getMessage(), 'Invalid device')
            ? "dev_$bad: OK\n"
            : "dev_$bad: BAD ({$e->getMessage()})\n";
    }
}

/* Non-numeric scalar inputs are rejected per arg. */
foreach (['stop' => [new stdClass(), 0, 1],
          'start' => [10, [1, 2], 1],
          'step' => [10, 0, null]] as $tag => [$a, $b, $c]) {
    try {
        NumPower::arange($a, $b, $c);
        echo "BAD: no throw on bad $tag\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'must be int, float, or numeric string')
            ? "bad_$tag: OK\n"
            : "bad_$tag: BAD ({$e->getMessage()})\n";
    }
}

/* uint64 with a negative numeric arg rejects with a clear message. */
foreach (['stop', 'start', 'step'] as $tag) {
    $args = [10, 0, 1];
    $idx  = ['stop' => 0, 'start' => 1, 'step' => 2][$tag];
    $args[$idx] = -1;
    try {
        NumPower::arange($args[0], $args[1], $args[2], 'uint64');
        echo "BAD: no throw on uint64 negative $tag\n";
    } catch (\Error $e) {
        echo str_contains($e->getMessage(), 'non-negative for uint64')
            ? "u64_neg_$tag: OK\n"
            : "u64_neg_$tag: BAD ({$e->getMessage()})\n";
    }
}

/* After every failure the engine still produces a correct arange. */
$ok = NumPower::arange(5);
echo 'recovered: ', ((string)$ok === '[0, 1, 2, 3, 4]' ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
step_zero: OK
nan: OK
dtype: OK
dev_2: OK
dev_-1: OK
dev_99: OK
bad_stop: OK
bad_start: OK
bad_step: OK
u64_neg_stop: OK
u64_neg_start: OK
u64_neg_step: OK
recovered: OK
