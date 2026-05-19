--TEST--
Stress-test arithmetic across all dtypes and devices in a loop — no VRAM leaks at RSHUTDOWN
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* If any device allocation isn't freed, RSHUTDOWN prints
   "VRAM MEMORY LEAK: leaked N array(s)". Asserting that this line does not
   appear is the actual leak check (handled by --EXPECT-- below). */
$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

for ($iter = 0; $iter < 5; $iter++) {
    foreach ($dtypes as $dt) {
        $n = NumPower::array([1, 0, 1, 2], $dt);
        $a = $n + 1;
        $b = $n + $a;
        $c = NumPower::add($a, $b);
        $c2 = NumPower::add($a, 2);
        $c3 = NumPower::add(2, $b);
        $d = $a->gpu();
        $e = $b->gpu();
        $f = $a + $b;
        $g = $a + 1;
        $h = 2 + $a;
        $i = 2 + $d;
        unset($n, $a, $b, $c, $c2, $c3, $d, $e, $f, $g, $h, $i);
    }
}
echo "OK\n";
?>
--EXPECT--
OK
