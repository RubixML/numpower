--TEST--
GPU unary ops (every dtype × every op × 50 iterations) leave no VRAM leaks at RSHUTDOWN
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); }
catch (\Error $e) { die("skip " . $e->getMessage()); }
?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Stress-test the typed-unary GPU dispatcher's memory bookkeeping:
   the per-op `NDArray_Copy` / `NDArray_AsType` allocations must be paired
   with NDArray_FREE on the input copy, and the typed kernel must not
   leave scratch buffers behind. Run 50 iterations of every op across
   every supported dtype; if a single byte is leaked, the RSHUTDOWN
   `vmemcheck` line prints `VRAM MEMORY LEAK: leaked N array(s)`. */

$dtypes_preserving = ['int8','uint8','int16','uint16','int32','uint32',
                       'int64','uint64','float16','float32','float64','float128'];
$dtypes_floats     = ['float16','float32','float64','float128'];

$data_int    = [-2, -1, 0, 1, 2, 3];
$data_uint   = [0, 1, 2, 3, 4, 5];
$data_float  = [-1.5, -0.5, 0.0, 0.5, 1.0, 2.0];
$data_fp128  = ['-1.5', '-0.5', '0', '0.5', '1', '2'];

for ($k = 0; $k < 50; $k++) {
    foreach ($dtypes_preserving as $dt) {
        $is_uint  = (strncmp($dt, 'uint', 4) === 0);
        $is_int   = (strncmp($dt, 'int',  3) === 0) || $is_uint;
        $is_fp128 = ($dt === 'float128');

        $raw = $is_fp128 ? $data_fp128 : ($is_uint ? $data_uint : ($is_int ? $data_int : $data_float));
        $a   = NumPower::array($raw, $dt)->gpu();

        $r1 = NumPower::abs($a);
        $r2 = NumPower::negative($a);
        $r3 = NumPower::positive($a);
        $r4 = NumPower::sign($a);
        $r5 = NumPower::square($a);
        $r6 = NumPower::clip($a, $is_fp128 ? '0' : 0,
                                 $is_fp128 ? '2' : 2);
        unset($r1, $r2, $r3, $r4, $r5, $r6, $a);
    }
    /* Float-only ops: sqrt / rsqrt / reciprocal / sinc */
    foreach ($dtypes_floats as $dt) {
        $is_fp128 = ($dt === 'float128');
        $raw = $is_fp128 ? ['1', '4', '9', '16'] : [1.0, 4.0, 9.0, 16.0];
        $a   = NumPower::array($raw, $dt)->gpu();
        $s1 = NumPower::sqrt($a);
        $s2 = NumPower::rsqrt($a);
        $s3 = NumPower::reciprocal($a);
        $s4 = NumPower::sinc($a);
        unset($s1, $s2, $s3, $s4, $a);
    }
    /* int → float promotion path on GPU */
    foreach (['int8','int32','int64','uint64'] as $dt) {
        $a = NumPower::array([1, 4, 9, 16], $dt)->gpu();
        $r = NumPower::sqrt($a);
        unset($r, $a);
    }
}
echo "DONE\n";
?>
--EXPECT--
DONE
