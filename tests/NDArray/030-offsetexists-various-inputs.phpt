--TEST--
NDArray::offsetExists() — isset($a[$offset]) is silent for any offset type, dtype-independent
--FILE--
<?php
/* PHP's standard ArrayAccess convention: isset() must NEVER throw, regardless
   of the offset's type. offsetExists() returns true only when the offset is
   an integer-coercible value within [0, shape[0]). Non-numeric offsets,
   negative indices, out-of-range indices, and 0-D source arrays all yield
   false silently — verified across CPU and every dtype. */

$dtypes = ['float4','float8','float16','float32','float64','float128',
           'int8','uint8','int16','uint16','int32','uint32','int64','uint64'];

foreach ($dtypes as $t) {
    $strIO = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $vals = $strIO ? ['1','2','3'] : [1, 2, 3];
    $a = new NDArray($vals, $t);

    /* Valid range. */
    $in0  = isset($a[0])    ? '1' : '0';
    $in2  = isset($a[2])    ? '1' : '0';
    /* Out of range. */
    $out3 = isset($a[3])    ? '1' : '0';
    $neg  = isset($a[-1])   ? '1' : '0';
    /* Non-integer offsets — must not throw. */
    $str_off = isset($a['x']) ? '1' : '0';
    $arr_off = isset($a[[0]]) ? '1' : '0';
    /* IS_DOUBLE coerces to long: 1.7 -> 1, in range -> true. */
    $dbl_in  = isset($a[1.7])  ? '1' : '0';
    /* IS_DOUBLE 99.0 -> 99, out of range -> false. */
    $dbl_out = isset($a[99.0]) ? '1' : '0';

    echo "$t: 0=$in0 2=$in2 3=$out3 -1=$neg 'x'=$str_off [0]=$arr_off 1.7=$dbl_in 99.0=$dbl_out\n";
}

/* 0-D source: every offset must be false. */
$s = new NDArray(7, 'float32');
$d0 = isset($s[0])    ? '1' : '0';
$dn = isset($s[-1])   ? '1' : '0';
$dx = isset($s['x'])  ? '1' : '0';
echo "0-D: 0=$d0 -1=$dn 'x'=$dx\n";

/* Edge cases that are non-integer-coercible — must all yield false, never
   crash, never throw. The NaN / Inf / overflow paths are particularly
   important: a naive (long)double cast would invoke UB and could land on
   LONG_MIN, masking the bug as "negative index". */
$a = new NDArray([1.0, 2.0, 3.0]);
$cases = [
    'null'          => null,
    'true'          => true,
    'false'         => false,
    'NaN'           => NAN,
    '+Inf'          => INF,
    '-Inf'          => -INF,
    'oversize-pos'  => 1e30,
    'oversize-neg'  => -1e30,
    'object'        => new stdClass(),
];
foreach ($cases as $name => $off) {
    $r = isset($a[$off]) ? '1' : '0';
    echo "edge $name=$r\n";
}

echo "done\n";
?>
--EXPECT--
float4: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
float8: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
float16: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
float32: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
float64: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
float128: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
int8: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
uint8: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
int16: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
uint16: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
int32: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
uint32: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
int64: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
uint64: 0=1 2=1 3=0 -1=0 'x'=0 [0]=0 1.7=1 99.0=0
0-D: 0=0 -1=0 'x'=0
edge null=0
edge true=0
edge false=0
edge NaN=0
edge +Inf=0
edge -Inf=0
edge oversize-pos=0
edge oversize-neg=0
edge object=0
done
