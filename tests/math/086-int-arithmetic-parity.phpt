--TEST--
PyTorch-parity integer arithmetic on every integer dtype (CPU + GPU)
--FILE--
<?php
/* PyTorch / NumPy convention: integer dtypes wrap modulo 2^bits on
   overflow (C semantics). The prior implementation promoted every integer
   dtype to float32 / float64 for the computation, which:
    - silently rounded `int32 * int32` products past 2^53 (e.g.
      `(2^28+1)^2` returned 536870912 instead of the PyTorch wrap of
      536870913),
    - diverged between CPU (double round-trip wraps) and GPU
      (`cuda_cast_f64_to_i32` saturates) on the same input. */

$gpu_available = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $gpu_available = false; }

function check($label, $got, $want) {
    if ((string)$got !== (string)$want) {
        echo "FAIL $label: got=", (string)$got, " want=$want\n";
        return false;
    }
    return true;
}

$ok = true;

/* INT8 wrap */
$a = new NDArray([127], 'int8');
$ok &= check('int8 INT8_MAX+1',  NumPower::add($a, 1)[0],     -128);
$ok &= check('int8 INT8_MIN-1',  NumPower::subtract(new NDArray([-128], 'int8'), 1)[0], 127);
$ok &= check('int8 64*4 wraps',  NumPower::multiply(new NDArray([64], 'int8'), 4)[0], 0);
$ok &= check('int8 2^7 wraps',   NumPower::pow(new NDArray([2], 'int8'), 7)[0], -128);

/* UINT8 wrap */
$ok &= check('uint8 255+1', NumPower::add(new NDArray([255], 'uint8'), 1)[0], 0);
$ok &= check('uint8 200*2', NumPower::multiply(new NDArray([200], 'uint8'), 2)[0], 144);

/* INT16 wrap */
$ok &= check('int16 INT16_MAX+1', NumPower::add(new NDArray([32767], 'int16'), 1)[0], -32768);
$ok &= check('int16 256*256',     NumPower::multiply(new NDArray([256], 'int16'), 256)[0], 0);

/* UINT16 wrap */
$ok &= check('uint16 UINT16_MAX+1', NumPower::add(new NDArray([65535], 'uint16'), 1)[0], 0);

/* INT32 wrap — the critical case where the prior float64 path diverged */
$ok &= check('int32 INT32_MAX+1',
             NumPower::add(new NDArray([2147483647], 'int32'), 1)[0], -2147483648);
/* (2^28+1)^2 = 2^56 + 2^29 + 1; low 32 bits = 2^29 + 1 = 536870913. */
$ok &= check('int32 (2^28+1)^2',
             NumPower::multiply(new NDArray([268435457], 'int32'),
                                new NDArray([268435457], 'int32'))[0],
             536870913);
$ok &= check('int32 10^10 wraps',
             NumPower::pow(new NDArray([10], 'int32'), 10)[0], 1410065408);

/* UINT32 wrap */
$ok &= check('uint32 UINT32_MAX+1',
             NumPower::add(new NDArray([4294967295], 'uint32'), 1)[0], 0);
$ok &= check('uint32 65536*65536 wraps',
             NumPower::multiply(new NDArray([65536], 'uint32'),
                                new NDArray([65536], 'uint32'))[0], 0);

/* INT64 wrap */
$ok &= check('int64 INT64_MAX+1',
             NumPower::add(new NDArray(['9223372036854775807'], 'int64'), 1)[0],
             '-9223372036854775808');
$ok &= check('int64 2^62',
             NumPower::pow(new NDArray([2], 'int64'), 62)[0],
             4611686018427387904);

/* UINT64 wrap */
$ok &= check('uint64 UINT64_MAX+1',
             NumPower::add(new NDArray(['18446744073709551615'], 'uint64'), 1)[0], '0');
$ok &= check('uint64 large+large preserved',
             NumPower::add(new NDArray(['18446744073709551610'], 'uint64'), '5')[0],
             '18446744073709551615');

/* CPU/GPU parity: same result on both devices */
if ($gpu_available) {
    foreach (['int8','uint8','int16','uint16','int32','uint32','int64','uint64'] as $dt) {
        $a = new NDArray([100, -50, 30], $dt);
        $b = new NDArray([2, 3, 5], $dt);
        $cpu_add = NumPower::add($a, $b)->toArray();
        $cpu_mul = NumPower::multiply($a, $b)->toArray();
        $cpu_sub = NumPower::subtract($a, $b)->toArray();
        $cpu_mod = NumPower::mod($a, $b)->toArray();
        $gpu_add = NumPower::add($a->gpu(), $b->gpu())->cpu()->toArray();
        $gpu_mul = NumPower::multiply($a->gpu(), $b->gpu())->cpu()->toArray();
        $gpu_sub = NumPower::subtract($a->gpu(), $b->gpu())->cpu()->toArray();
        $gpu_mod = NumPower::mod($a->gpu(), $b->gpu())->cpu()->toArray();
        foreach (['add' => [$cpu_add, $gpu_add],
                  'mul' => [$cpu_mul, $gpu_mul],
                  'sub' => [$cpu_sub, $gpu_sub],
                  'mod' => [$cpu_mod, $gpu_mod]] as $op => $pair) {
            if (json_encode($pair[0]) !== json_encode($pair[1])) {
                echo "FAIL $dt $op CPU=", json_encode($pair[0]),
                     " GPU=", json_encode($pair[1]), "\n";
                $ok = false;
            }
        }
    }
}

/* Integer mod uses sign-of-dividend (C semantics, matches `torch.fmod`). */
$ok &= check('i32 -7 mod 3', NumPower::mod(new NDArray([-7], 'int32'), 3)[0], -1);
$ok &= check('i32 7 mod -3', NumPower::mod(new NDArray([7],  'int32'), -3)[0], 1);
/* Divide-by-zero on int → 0 (avoids SIGFPE). */
$ok &= check('i32 10 mod 0', NumPower::mod(new NDArray([10], 'int32'), 0)[0], 0);

echo $ok ? "ok\n" : "FAIL\n";
?>
--EXPECT--
ok
