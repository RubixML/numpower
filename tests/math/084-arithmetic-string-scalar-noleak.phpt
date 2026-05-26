--TEST--
String-scalar arithmetic + wide-dtype reductions don't leak VRAM at RSHUTDOWN
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Two newly-introduced paths exercised in a loop while NDARRAY_VCHECK=1 is
   on the parent process: (1) the string-scalar weak-promotion path which
   allocates a transient 0-D NDArray per call and (2) the axis-reduction
   GPU path which rolls + casts + per-slice accumulates. Both must release
   every intermediate vmalloc slot or vmemcheck() at RSHUTDOWN will report
   a leak. */

putenv('NDARRAY_VCHECK=1');

$ops = ['add', 'subtract', 'multiply', 'mod', 'pow'];
$dtypes = ['int64', 'uint64', 'float128'];

foreach ($dtypes as $t) {
    for ($i = 0; $i < 20; $i++) {
        $a = (new NDArray(['100', '200', '300'], $t))->gpu();
        $r1 = NumPower::add($a, '50');           /* string scalar + GPU array */
        $r2 = NumPower::multiply($a, '2');
        $r3 = NumPower::sum($a);                  /* full reduction */
        $r4 = NumPower::sum($a, 0);               /* axis reduction (1-D → 0-D collapse) */
        $r5 = NumPower::prod($a, 0);
        /* Combine ops to make sure mid-pipeline NDArrays are freed. */
        $r6 = NumPower::add(NumPower::multiply($a, '3'), '7');
        unset($a, $r1, $r2, $r3, $r4, $r5, $r6);
    }
}

/* Axis reductions on a 2-D GPU array exercise the cast+roll+kernel loop. */
foreach ($dtypes as $t) {
    for ($i = 0; $i < 20; $i++) {
        $m = (new NDArray([['10','20','30'],['40','50','60']], $t))->gpu();
        $r0 = NumPower::sum($m, 0);
        $r1 = NumPower::sum($m, 1);
        $r2 = NumPower::prod($m, 0);
        $r3 = NumPower::prod($m, -1);
        unset($m, $r0, $r1, $r2, $r3);
    }
}

/* Stress the narrow-int axis path (cast to int64 / uint64 on GPU). */
foreach (['int8','uint8','int16','uint16','int32','uint32'] as $t) {
    for ($i = 0; $i < 10; $i++) {
        $m = (new NDArray([[1,2,3],[4,5,6],[7,8,9]], $t))->gpu();
        $r = NumPower::sum($m, 0);
        $r2 = NumPower::sum($m, 1);
        $r3 = NumPower::prod($m, -1);
        unset($m, $r, $r2, $r3);
    }
}

/* fp4 / fp8 axis path goes through float16 staging on GPU. */
foreach (['float4','float8'] as $t) {
    for ($i = 0; $i < 10; $i++) {
        $m = (new NDArray([[1,2,3],[4,5,6]], $t))->gpu();
        $r0 = NumPower::sum($m, 0);
        $r1 = NumPower::prod($m, 1);
        unset($m, $r0, $r1);
    }
}

echo "ok\n";
?>
--EXPECT--
ok
