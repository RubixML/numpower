--TEST--
Arithmetic on empty NDArrays: shape preserved, dtype promoted, no garbage values
--FILE--
<?php
/* Empty-broadcast semantics (NumPy / PyTorch parity):
     (0,)   broadcast (1,)   -> (0,)
     (1,)   broadcast (0,)   -> (0,)
     (0,)   broadcast (0,)   -> (0,)
     (0, N) broadcast (1, N) -> (0, N)
     (0,)   broadcast (5,)   -> ValueError
   The result must carry the broadcast shape, the promoted dtype, and an empty
   data buffer — never a garbage 1-element scalar (the old broken behaviour
   that surfaced as `echo $a` printing `[5.0000014]`). */

function ser(NDArray $a): array {
    $s = $a->__serialize();
    return [$a->shape(), $s['dtype'], $s['data']];
}

/* Exact dtype is set in the bug repro: __construct(array) defaults to float32. */
echo "=== bug reproducer (echo \$t + new NDArray([5])) ===\n";
$t = new NDArray([]);
$a = $t + new NDArray([5]);
[$sh, $dt, $data] = ser($a);
echo "shape=", json_encode($sh), " dtype=$dt data=", json_encode($data), " toString=", $a, "\n";

echo "\n=== every dtype × every op: (0,) op (1,) ===\n";
/* Cover all 14 dtypes the extension supports. The exact data buffer must be []
   (empty) and the broadcast shape (0,) regardless of dtype. ZEND_DIV ('/')
   upgrades integer dtypes to float per PyTorch true-division: narrow ints
   (8/16-bit) -> float32, 32/64-bit ints -> float64. Every other op preserves
   the input dtype. */
$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];
$ops = ['+', '-', '*', '/', '**'];

/* dtype after `/` per PyTorch rule. Same-dtype on both sides, so promote is a no-op. */
function div_dtype(string $dt): string {
    static $upgrade = [
        'int8'=>'float32','uint8'=>'float32','int16'=>'float32','uint16'=>'float32',
        'int32'=>'float64','uint32'=>'float64','int64'=>'float64','uint64'=>'float64',
    ];
    return $upgrade[$dt] ?? $dt;
}

foreach ($dtypes as $dt) {
    foreach ($ops as $op) {
        $e = NumPower::array([],  $dt);
        $b = NumPower::array([5], $dt);
        $r = eval("return \$e $op \$b;");
        [$sh, $rdt, $data] = ser($r);
        $want_dt = ($op === '/') ? div_dtype($dt) : $dt;
        $ok = $sh === [0] && $rdt === $want_dt && $data === [];
        echo str_pad($dt, 10), " ", str_pad($op, 2),
             " shape=", json_encode($sh),
             " dtype=", str_pad($rdt, 9),
             " data=", json_encode($data),
             " ", ($ok ? 'OK' : 'BAD'),
             "\n";
    }
}

echo "\n=== other-direction & both-empty ===\n";
/* Same broadcast result, reversed operand order. Confirms the short-circuit
   triggers regardless of which side is empty. */
$cases = [
    '(1,) + (0,)' => [new NDArray([5]), new NDArray([])],
    '(0,) + (0,)' => [new NDArray([]),  new NDArray([])],
];
foreach ($cases as $label => [$a, $b]) {
    $r = $a + $b;
    [$sh, $dt, $data] = ser($r);
    echo "$label shape=", json_encode($sh), " dtype=$dt data=", json_encode($data), "\n";
}

echo "\n=== 2-D broadcast: (0,3) op (1,3) -> (0,3) ===\n";
/* zeros() default dtype is float32. Multi-dim empty case validates that the
   broadcast-shape helper aligns dims from the right and lets 0 win over 1. */
foreach ($ops as $op) {
    $z = NumPower::zeros([0, 3]);
    $o = NumPower::ones([1, 3]);
    $r = eval("return \$z $op \$o;");
    [$sh, $dt, $data] = ser($r);
    $want_dt = ($op === '/') ? 'float32' : 'float32';
    $ok = $sh === [0, 3] && $dt === $want_dt && $data === [];
    echo "(0,3) $op (1,3) shape=", json_encode($sh), " dtype=$dt ", ($ok ? 'OK' : 'BAD'), "\n";
}

echo "\n=== mixed-dtype promotion on empty ===\n";
/* Promotion rules from promote_dtype() must still apply for empty operands.
     int8 + int16 -> int16
     int8 + uint8 -> int16 (PyTorch rule)
     int32 + float32 -> float32 (float beats int)
     float16 + float32 -> float32 (wider float wins)
     uint32 + int32 -> int64
*/
$mixed = [
    ['int8','int16',  '+', 'int16'],
    ['int8','uint8',  '+', 'int16'],
    ['int32','float32','+', 'float32'],
    ['float16','float32','+', 'float32'],
    ['uint32','int32', '+', 'int64'],
];
foreach ($mixed as [$da, $db, $op, $want]) {
    $e = NumPower::array([], $da);
    $f = NumPower::array([1], $db);
    $r = eval("return \$e $op \$f;");
    [$sh, $dt, $data] = ser($r);
    $ok = $sh === [0] && $dt === $want && $data === [];
    echo "$da $op $db -> dtype=$dt want=$want ", ($ok ? 'OK' : 'BAD'), "\n";
}

echo "\n=== mod on empty (int and float) ===\n";
/* NumPower::mod follows ZEND_MOD: kernels are float-only internally but the
   result is cast back to the promoted dtype, so int % int stays int. */
foreach (['int32', 'float32', 'float64'] as $dt) {
    $e = NumPower::array([],  $dt);
    $b = NumPower::array([5], $dt);
    $r = NumPower::mod($e, $b);
    [$sh, $rdt, $data] = ser($r);
    $ok = $sh === [0] && $rdt === $dt && $data === [];
    echo "$dt mod -> dtype=$rdt ", ($ok ? 'OK' : 'BAD'), "\n";
}

echo "\n=== empty + 0-D scalar and empty + PHP scalar ===\n";
/* The 0-D scalar path goes through the scalar-reshape branch in
   ndarray_promote_and_op (NDIM(b)==0). The PHP scalar path constructs a 1-D
   weak scalar in ndarray_do_operation_ex before reaching the kernel. Both
   must collapse to the empty broadcast shape, not allocate stray output. */
$s = NumPower::array(7);
$r = (new NDArray([])) + $s;
[$sh, $dt, $data] = ser($r);
echo "empty + 0-D scalar shape=", json_encode($sh), " data=", json_encode($data), "\n";

$r = (new NDArray([])) + 5;
[$sh, $dt, $data] = ser($r);
echo "empty + 5 (int)    shape=", json_encode($sh), " dtype=$dt data=", json_encode($data), "\n";

$r = 5 + (new NDArray([]));
[$sh, $dt, $data] = ser($r);
echo "5 + empty (int)    shape=", json_encode($sh), " dtype=$dt data=", json_encode($data), "\n";

$r = (new NDArray([])) + 5.5;
[$sh, $dt, $data] = ser($r);
echo "empty + 5.5 (float)shape=", json_encode($sh), " dtype=$dt data=", json_encode($data), "\n";

echo "\n=== incompatible empty shapes still throw ===\n";
/* IsBroadcastable still rejects shapes that don't have a 1 to broadcast to:
   (0,) vs (5,) and (0,3) vs (2,3) are both errors, just like NumPy. */
try {
    $r = (new NDArray([])) + (new NDArray([1, 2, 3, 4, 5]));
    echo "(0,)+(5,) MISSING_EXC\n";
} catch (\Error $e) {
    echo "(0,)+(5,) throws: ", $e->getMessage(), "\n";
}
try {
    $r = NumPower::zeros([0, 3]) + NumPower::ones([2, 3]);
    echo "(0,3)+(2,3) MISSING_EXC\n";
} catch (\Error $e) {
    echo "(0,3)+(2,3) throws: ", $e->getMessage(), "\n";
}

echo "\n=== boundary: result of empty op is operand-safe ===\n";
/* The short-circuit must not leak the result's data pointer or shape buffer.
   Stress it with a loop that builds and discards 1000 empty arithmetic
   results; if either side leaks, the PHP heap blows up on shutdown. */
for ($i = 0; $i < 1000; $i++) {
    $r = (new NDArray([])) + (new NDArray([$i % 7]));
}
echo "1000-iter stress: ok\n";
?>
--EXPECTF--
=== bug reproducer (echo $t + new NDArray([5])) ===
shape=[0] dtype=float32 data=[] toString=[]

=== every dtype × every op: (0,) op (1,) ===
float4     +  shape=[0] dtype=float4    data=[] OK
float4     -  shape=[0] dtype=float4    data=[] OK
float4     *  shape=[0] dtype=float4    data=[] OK
float4     /  shape=[0] dtype=float4    data=[] OK
float4     ** shape=[0] dtype=float4    data=[] OK
float8     +  shape=[0] dtype=float8    data=[] OK
float8     -  shape=[0] dtype=float8    data=[] OK
float8     *  shape=[0] dtype=float8    data=[] OK
float8     /  shape=[0] dtype=float8    data=[] OK
float8     ** shape=[0] dtype=float8    data=[] OK
float16    +  shape=[0] dtype=float16   data=[] OK
float16    -  shape=[0] dtype=float16   data=[] OK
float16    *  shape=[0] dtype=float16   data=[] OK
float16    /  shape=[0] dtype=float16   data=[] OK
float16    ** shape=[0] dtype=float16   data=[] OK
float32    +  shape=[0] dtype=float32   data=[] OK
float32    -  shape=[0] dtype=float32   data=[] OK
float32    *  shape=[0] dtype=float32   data=[] OK
float32    /  shape=[0] dtype=float32   data=[] OK
float32    ** shape=[0] dtype=float32   data=[] OK
float64    +  shape=[0] dtype=float64   data=[] OK
float64    -  shape=[0] dtype=float64   data=[] OK
float64    *  shape=[0] dtype=float64   data=[] OK
float64    /  shape=[0] dtype=float64   data=[] OK
float64    ** shape=[0] dtype=float64   data=[] OK
float128   +  shape=[0] dtype=float128  data=[] OK
float128   -  shape=[0] dtype=float128  data=[] OK
float128   *  shape=[0] dtype=float128  data=[] OK
float128   /  shape=[0] dtype=float128  data=[] OK
float128   ** shape=[0] dtype=float128  data=[] OK
int8       +  shape=[0] dtype=int8      data=[] OK
int8       -  shape=[0] dtype=int8      data=[] OK
int8       *  shape=[0] dtype=int8      data=[] OK
int8       /  shape=[0] dtype=float32   data=[] OK
int8       ** shape=[0] dtype=int8      data=[] OK
uint8      +  shape=[0] dtype=uint8     data=[] OK
uint8      -  shape=[0] dtype=uint8     data=[] OK
uint8      *  shape=[0] dtype=uint8     data=[] OK
uint8      /  shape=[0] dtype=float32   data=[] OK
uint8      ** shape=[0] dtype=uint8     data=[] OK
int16      +  shape=[0] dtype=int16     data=[] OK
int16      -  shape=[0] dtype=int16     data=[] OK
int16      *  shape=[0] dtype=int16     data=[] OK
int16      /  shape=[0] dtype=float32   data=[] OK
int16      ** shape=[0] dtype=int16     data=[] OK
uint16     +  shape=[0] dtype=uint16    data=[] OK
uint16     -  shape=[0] dtype=uint16    data=[] OK
uint16     *  shape=[0] dtype=uint16    data=[] OK
uint16     /  shape=[0] dtype=float32   data=[] OK
uint16     ** shape=[0] dtype=uint16    data=[] OK
int32      +  shape=[0] dtype=int32     data=[] OK
int32      -  shape=[0] dtype=int32     data=[] OK
int32      *  shape=[0] dtype=int32     data=[] OK
int32      /  shape=[0] dtype=float64   data=[] OK
int32      ** shape=[0] dtype=int32     data=[] OK
uint32     +  shape=[0] dtype=uint32    data=[] OK
uint32     -  shape=[0] dtype=uint32    data=[] OK
uint32     *  shape=[0] dtype=uint32    data=[] OK
uint32     /  shape=[0] dtype=float64   data=[] OK
uint32     ** shape=[0] dtype=uint32    data=[] OK
int64      +  shape=[0] dtype=int64     data=[] OK
int64      -  shape=[0] dtype=int64     data=[] OK
int64      *  shape=[0] dtype=int64     data=[] OK
int64      /  shape=[0] dtype=float64   data=[] OK
int64      ** shape=[0] dtype=int64     data=[] OK
uint64     +  shape=[0] dtype=uint64    data=[] OK
uint64     -  shape=[0] dtype=uint64    data=[] OK
uint64     *  shape=[0] dtype=uint64    data=[] OK
uint64     /  shape=[0] dtype=float64   data=[] OK
uint64     ** shape=[0] dtype=uint64    data=[] OK

=== other-direction & both-empty ===
(1,) + (0,) shape=[0] dtype=float32 data=[]
(0,) + (0,) shape=[0] dtype=float32 data=[]

=== 2-D broadcast: (0,3) op (1,3) -> (0,3) ===
(0,3) + (1,3) shape=[0,3] dtype=float32 OK
(0,3) - (1,3) shape=[0,3] dtype=float32 OK
(0,3) * (1,3) shape=[0,3] dtype=float32 OK
(0,3) / (1,3) shape=[0,3] dtype=float32 OK
(0,3) ** (1,3) shape=[0,3] dtype=float32 OK

=== mixed-dtype promotion on empty ===
int8 + int16 -> dtype=int16 want=int16 OK
int8 + uint8 -> dtype=int16 want=int16 OK
int32 + float32 -> dtype=float32 want=float32 OK
float16 + float32 -> dtype=float32 want=float32 OK
uint32 + int32 -> dtype=int64 want=int64 OK

=== mod on empty (int and float) ===
int32 mod -> dtype=int32 OK
float32 mod -> dtype=float32 OK
float64 mod -> dtype=float64 OK

=== empty + 0-D scalar and empty + PHP scalar ===
empty + 0-D scalar shape=[0] data=[]
empty + 5 (int)    shape=[0] dtype=float32 data=[]
5 + empty (int)    shape=[0] dtype=float32 data=[]
empty + 5.5 (float)shape=[0] dtype=float32 data=[]

=== incompatible empty shapes still throw ===
(0,)+(5,) throws: %s
(0,3)+(2,3) throws: %s

=== boundary: result of empty op is operand-safe ===
1000-iter stress: ok
