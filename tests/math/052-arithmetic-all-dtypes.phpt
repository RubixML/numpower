--TEST--
Arithmetic ops (+,-,*,/,**) on every dtype: same-type arr+arr preserves dtype, scalar+arr preserves dtype (PyTorch weak-scalar rule)
--FILE--
<?php
/* Verify that:
   - $a + $b for same-dtype arrays computes correct values for every dtype,
   - $a + scalar preserves $a's dtype (PyTorch weak scalar promotion),
   - the result element type via $r[i] matches the dtype's native PHP type:
     int*  / uint*  (except uint64) → int
     float4..float64                → float
     float128, uint64               → string
*/
function phptype($v) {
    if (is_int($v))    return 'int';
    if (is_float($v))  return 'float';
    if (is_string($v)) return 'string';
    return gettype($v);
}

$expected_phptype = [
    'float4'   => 'float',  'float8'   => 'float',  'float16'  => 'float',
    'float32'  => 'float',  'float64'  => 'float',  'float128' => 'string',
    'int8'     => 'int',    'uint8'    => 'int',
    'int16'    => 'int',    'uint16'   => 'int',
    'int32'    => 'int',    'uint32'   => 'int',
    'int64'    => 'int',    'uint64'   => 'string',
];

$dtypes = array_keys($expected_phptype);

foreach ($dtypes as $dt) {
    /* values fit every integer dtype and are exact in every float dtype */
    $a = NumPower::array([2, 2, 2, 2], $dt);
    $b = NumPower::array([3, 3, 3, 3], $dt);

    /* arr+arr */
    $r = $a + $b;
    $want_type = $expected_phptype[$dt];
    $got_type  = phptype($r[0]);
    /* float4 (E2M1) rounds 5 → 4 (nearest representable). other dtypes give 5. */
    $expect_val = ($dt === 'float4') ? '4' : '5';
    $got_val    = (string)$r[0];
    echo "$dt arr+arr type=$got_type want=$want_type val=$got_val expect=$expect_val ",
         ($got_type === $want_type && $got_val === $expect_val ? 'OK' : 'BAD'), "\n";

    /* scalar+arr preserves dtype (weak-scalar) */
    $r2 = $a + 3;
    $got_type2 = phptype($r2[0]);
    $got_val2  = (string)$r2[0];
    /* Same arithmetic, same expected value */
    echo "$dt arr+scalar type=$got_type2 want=$want_type val=$got_val2 expect=$expect_val ",
         ($got_type2 === $want_type && $got_val2 === $expect_val ? 'OK' : 'BAD'), "\n";

    /* scalar+arr (LHS scalar) */
    $r3 = 3 + $a;
    $got_type3 = phptype($r3[0]);
    $got_val3  = (string)$r3[0];
    echo "$dt scalar+arr type=$got_type3 want=$want_type val=$got_val3 expect=$expect_val ",
         ($got_type3 === $want_type && $got_val3 === $expect_val ? 'OK' : 'BAD'), "\n";
}

/* All operations: same-dtype +, -, *, /, ** for float64 */
$a = NumPower::array([6, 6, 6, 6], 'float64');
$b = NumPower::array([2, 2, 2, 2], 'float64');
echo "+: ", ($a + $b)[0], "\n";
echo "-: ", ($a - $b)[0], "\n";
echo "*: ", ($a * $b)[0], "\n";
echo "/: ", ($a / $b)[0], "\n";
echo "**: ", ($a ** $b)[0], "\n";

/* True division on integer inputs promotes to float (PyTorch). */
$ai = NumPower::array([5, 5, 5, 5], 'int32');
$d  = $ai / 2;
echo "int32 / 2 type=", phptype($d[0]), " val=", (string)$d[0], "\n";
?>
--EXPECT--
float4 arr+arr type=float want=float val=4 expect=4 OK
float4 arr+scalar type=float want=float val=4 expect=4 OK
float4 scalar+arr type=float want=float val=4 expect=4 OK
float8 arr+arr type=float want=float val=5 expect=5 OK
float8 arr+scalar type=float want=float val=5 expect=5 OK
float8 scalar+arr type=float want=float val=5 expect=5 OK
float16 arr+arr type=float want=float val=5 expect=5 OK
float16 arr+scalar type=float want=float val=5 expect=5 OK
float16 scalar+arr type=float want=float val=5 expect=5 OK
float32 arr+arr type=float want=float val=5 expect=5 OK
float32 arr+scalar type=float want=float val=5 expect=5 OK
float32 scalar+arr type=float want=float val=5 expect=5 OK
float64 arr+arr type=float want=float val=5 expect=5 OK
float64 arr+scalar type=float want=float val=5 expect=5 OK
float64 scalar+arr type=float want=float val=5 expect=5 OK
float128 arr+arr type=string want=string val=5 expect=5 OK
float128 arr+scalar type=string want=string val=5 expect=5 OK
float128 scalar+arr type=string want=string val=5 expect=5 OK
int8 arr+arr type=int want=int val=5 expect=5 OK
int8 arr+scalar type=int want=int val=5 expect=5 OK
int8 scalar+arr type=int want=int val=5 expect=5 OK
uint8 arr+arr type=int want=int val=5 expect=5 OK
uint8 arr+scalar type=int want=int val=5 expect=5 OK
uint8 scalar+arr type=int want=int val=5 expect=5 OK
int16 arr+arr type=int want=int val=5 expect=5 OK
int16 arr+scalar type=int want=int val=5 expect=5 OK
int16 scalar+arr type=int want=int val=5 expect=5 OK
uint16 arr+arr type=int want=int val=5 expect=5 OK
uint16 arr+scalar type=int want=int val=5 expect=5 OK
uint16 scalar+arr type=int want=int val=5 expect=5 OK
int32 arr+arr type=int want=int val=5 expect=5 OK
int32 arr+scalar type=int want=int val=5 expect=5 OK
int32 scalar+arr type=int want=int val=5 expect=5 OK
uint32 arr+arr type=int want=int val=5 expect=5 OK
uint32 arr+scalar type=int want=int val=5 expect=5 OK
uint32 scalar+arr type=int want=int val=5 expect=5 OK
int64 arr+arr type=int want=int val=5 expect=5 OK
int64 arr+scalar type=int want=int val=5 expect=5 OK
int64 scalar+arr type=int want=int val=5 expect=5 OK
uint64 arr+arr type=string want=string val=5 expect=5 OK
uint64 arr+scalar type=string want=string val=5 expect=5 OK
uint64 scalar+arr type=string want=string val=5 expect=5 OK
+: 8
-: 4
*: 12
/: 3
**: 36
int32 / 2 type=float val=2.5
