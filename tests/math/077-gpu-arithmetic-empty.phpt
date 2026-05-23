--TEST--
GPU arithmetic on empty NDArrays: every dtype produces an empty GPU result, no kernel UB
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* GPU mirror of 076-arithmetic-empty.phpt. The short-circuit lives in
   ndarray_promote_and_op above the GPU dispatch, so empty operands never
   reach a GPU kernel — but the resulting empty NDArray must still be on
   the requested device. */

function ser_cpu(NDArray $a): array {
    $cpu = $a->cpu();
    $s = $cpu->__serialize();
    return [$cpu->shape(), $s['dtype'], $s['data']];
}

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

function div_dtype(string $dt): string {
    static $upgrade = [
        'int8'=>'float32','uint8'=>'float32','int16'=>'float32','uint16'=>'float32',
        'int32'=>'float64','uint32'=>'float64','int64'=>'float64','uint64'=>'float64',
    ];
    return $upgrade[$dt] ?? $dt;
}

echo "=== GPU empty op non-empty: every dtype × every op ===\n";
$ops = ['+', '-', '*', '/', '**'];
foreach ($dtypes as $dt) {
    foreach ($ops as $op) {
        $e = NumPower::array([],  $dt)->gpu();
        $b = NumPower::array([5], $dt)->gpu();
        $r = eval("return \$e $op \$b;");
        $stays_on_gpu = $r->isGPU();
        [$sh, $rdt, $data] = ser_cpu($r);
        $want_dt = ($op === '/') ? div_dtype($dt) : $dt;
        $ok = $sh === [0] && $rdt === $want_dt && $data === [] && $stays_on_gpu;
        echo str_pad($dt, 10), " ", str_pad($op, 2),
             " gpu=", ($stays_on_gpu ? 'yes' : 'no'),
             " shape=", json_encode($sh),
             " dtype=", str_pad($rdt, 9),
             " ", ($ok ? 'OK' : 'BAD'),
             "\n";
    }
}

echo "\n=== GPU 2-D broadcast: (0,3) + (1,3) ===\n";
/* zeros() returns CPU; bounce to GPU via ->gpu() so both sides match. */
$z = NumPower::zeros([0, 3])->gpu();
$o = NumPower::ones([1, 3])->gpu();
$r = $z + $o;
[$sh, $dt, $data] = ser_cpu($r);
$ok = $sh === [0, 3] && $dt === 'float32' && $data === [] && $r->isGPU();
echo "(0,3)+(1,3) gpu=", ($r->isGPU() ? 'yes' : 'no'),
     " shape=", json_encode($sh), " dtype=$dt ", ($ok ? 'OK' : 'BAD'), "\n";

echo "\n=== GPU device mismatch on non-scalar empty throws ===\n";
/* Empty array is NDIM=1 (not a 0-D scalar) so the device-mismatch check in
   ndarray_promote_and_op must reject GPU empty + CPU non-empty, matching
   PyTorch's "Expected all tensors to be on the same device" rule. */
try {
    $r = NumPower::array([], 'float32')->gpu()
       + NumPower::array([5], 'float32');
    echo "MISSING_EXC\n";
} catch (\Error $e) {
    echo "throws: ", $e->getMessage(), "\n";
}

echo "\n=== GPU 0-D scalar + empty(GPU) migrates scalar ===\n";
/* A 0-D scalar is migrated to the other operand's device by the
   device-handling block. The migrated scalar + empty must still short-circuit
   to an empty result on the GPU. */
$s = NumPower::array(7);                /* CPU 0-D scalar */
$e = NumPower::array([], 'float32')->gpu();
$r = $e + $s;
[$sh, $dt, $data] = ser_cpu($r);
$ok = $sh === [0] && $dt === 'float32' && $data === [] && $r->isGPU();
echo "GPU empty + CPU 0-D scalar gpu=", ($r->isGPU() ? 'yes' : 'no'),
     " shape=", json_encode($sh), " ", ($ok ? 'OK' : 'BAD'), "\n";

echo "\n=== GPU empty stress (no VRAM leak) ===\n";
/* Repeat the empty short-circuit 200 times to confirm we don't leak GPU
   buffers — every iteration's NDArray_Empty(device=GPU) allocates a 0-byte
   VRAM block that must be released when the result goes out of scope. */
for ($i = 0; $i < 200; $i++) {
    $r = (NumPower::array([], 'float32')->gpu())
       + (NumPower::array([$i % 7], 'float32')->gpu());
}
echo "200-iter GPU stress: ok\n";
?>
--EXPECTF--
=== GPU empty op non-empty: every dtype × every op ===
float4     +  gpu=yes shape=[0] dtype=float4    OK
float4     -  gpu=yes shape=[0] dtype=float4    OK
float4     *  gpu=yes shape=[0] dtype=float4    OK
float4     /  gpu=yes shape=[0] dtype=float4    OK
float4     ** gpu=yes shape=[0] dtype=float4    OK
float8     +  gpu=yes shape=[0] dtype=float8    OK
float8     -  gpu=yes shape=[0] dtype=float8    OK
float8     *  gpu=yes shape=[0] dtype=float8    OK
float8     /  gpu=yes shape=[0] dtype=float8    OK
float8     ** gpu=yes shape=[0] dtype=float8    OK
float16    +  gpu=yes shape=[0] dtype=float16   OK
float16    -  gpu=yes shape=[0] dtype=float16   OK
float16    *  gpu=yes shape=[0] dtype=float16   OK
float16    /  gpu=yes shape=[0] dtype=float16   OK
float16    ** gpu=yes shape=[0] dtype=float16   OK
float32    +  gpu=yes shape=[0] dtype=float32   OK
float32    -  gpu=yes shape=[0] dtype=float32   OK
float32    *  gpu=yes shape=[0] dtype=float32   OK
float32    /  gpu=yes shape=[0] dtype=float32   OK
float32    ** gpu=yes shape=[0] dtype=float32   OK
float64    +  gpu=yes shape=[0] dtype=float64   OK
float64    -  gpu=yes shape=[0] dtype=float64   OK
float64    *  gpu=yes shape=[0] dtype=float64   OK
float64    /  gpu=yes shape=[0] dtype=float64   OK
float64    ** gpu=yes shape=[0] dtype=float64   OK
float128   +  gpu=yes shape=[0] dtype=float128  OK
float128   -  gpu=yes shape=[0] dtype=float128  OK
float128   *  gpu=yes shape=[0] dtype=float128  OK
float128   /  gpu=yes shape=[0] dtype=float128  OK
float128   ** gpu=yes shape=[0] dtype=float128  OK
int8       +  gpu=yes shape=[0] dtype=int8      OK
int8       -  gpu=yes shape=[0] dtype=int8      OK
int8       *  gpu=yes shape=[0] dtype=int8      OK
int8       /  gpu=yes shape=[0] dtype=float32   OK
int8       ** gpu=yes shape=[0] dtype=int8      OK
uint8      +  gpu=yes shape=[0] dtype=uint8     OK
uint8      -  gpu=yes shape=[0] dtype=uint8     OK
uint8      *  gpu=yes shape=[0] dtype=uint8     OK
uint8      /  gpu=yes shape=[0] dtype=float32   OK
uint8      ** gpu=yes shape=[0] dtype=uint8     OK
int16      +  gpu=yes shape=[0] dtype=int16     OK
int16      -  gpu=yes shape=[0] dtype=int16     OK
int16      *  gpu=yes shape=[0] dtype=int16     OK
int16      /  gpu=yes shape=[0] dtype=float32   OK
int16      ** gpu=yes shape=[0] dtype=int16     OK
uint16     +  gpu=yes shape=[0] dtype=uint16    OK
uint16     -  gpu=yes shape=[0] dtype=uint16    OK
uint16     *  gpu=yes shape=[0] dtype=uint16    OK
uint16     /  gpu=yes shape=[0] dtype=float32   OK
uint16     ** gpu=yes shape=[0] dtype=uint16    OK
int32      +  gpu=yes shape=[0] dtype=int32     OK
int32      -  gpu=yes shape=[0] dtype=int32     OK
int32      *  gpu=yes shape=[0] dtype=int32     OK
int32      /  gpu=yes shape=[0] dtype=float64   OK
int32      ** gpu=yes shape=[0] dtype=int32     OK
uint32     +  gpu=yes shape=[0] dtype=uint32    OK
uint32     -  gpu=yes shape=[0] dtype=uint32    OK
uint32     *  gpu=yes shape=[0] dtype=uint32    OK
uint32     /  gpu=yes shape=[0] dtype=float64   OK
uint32     ** gpu=yes shape=[0] dtype=uint32    OK
int64      +  gpu=yes shape=[0] dtype=int64     OK
int64      -  gpu=yes shape=[0] dtype=int64     OK
int64      *  gpu=yes shape=[0] dtype=int64     OK
int64      /  gpu=yes shape=[0] dtype=float64   OK
int64      ** gpu=yes shape=[0] dtype=int64     OK
uint64     +  gpu=yes shape=[0] dtype=uint64    OK
uint64     -  gpu=yes shape=[0] dtype=uint64    OK
uint64     *  gpu=yes shape=[0] dtype=uint64    OK
uint64     /  gpu=yes shape=[0] dtype=float64   OK
uint64     ** gpu=yes shape=[0] dtype=uint64    OK

=== GPU 2-D broadcast: (0,3) + (1,3) ===
(0,3)+(1,3) gpu=yes shape=[0,3] dtype=float32 OK

=== GPU device mismatch on non-scalar empty throws ===
throws: %s

=== GPU 0-D scalar + empty(GPU) migrates scalar ===
GPU empty + CPU 0-D scalar gpu=yes shape=[0] OK

=== GPU empty stress (no VRAM leak) ===
200-iter GPU stress: ok
