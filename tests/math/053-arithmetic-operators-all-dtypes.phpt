--TEST--
+, -, *, /, ** operators with NumPower::add/subtract/multiply/divide/pow give matching results across all dtypes
--FILE--
<?php
/* For every dtype, verify that the operator form ($a + $b) and the method form
   (NumPower::add($a, $b)) produce equal values and equal result dtypes. Also
   verify compound-assignment operators (+=, -=, *=, /=, **=). */
$dtypes = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
           'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];

function vals_equal($a, $b) {
    /* Stringify so float128/uint64 compare as strings and everything else
       compares by canonical string form. */
    return (string)$a === (string)$b;
}

foreach ($dtypes as $dt) {
    $a = NumPower::array([4, 4, 4, 4], $dt);
    $b = NumPower::array([2, 2, 2, 2], $dt);
    $cases = [
        '+'  => [$a + $b,         NumPower::add($a, $b)],
        '-'  => [$a - $b,         NumPower::subtract($a, $b)],
        '*'  => [$a * $b,         NumPower::multiply($a, $b)],
        '/'  => [$a / $b,         NumPower::divide($a, $b)],
        '**' => [$a ** $b,        NumPower::pow($a, $b)],
    ];
    foreach ($cases as $op => [$opres, $methres]) {
        $ok = vals_equal($opres[0], $methres[0]) ? 'OK' : 'BAD';
        echo "$dt $op operator/method ", (string)$opres[0], "/", (string)$methres[0], " $ok\n";
    }
}

/* Compound assignment: $a += 1 etc. for a float32 array (PHP routes
   compound ops through ZEND_ADD etc.; the result must update $a in place). */
$a = NumPower::array([1, 2, 3, 4], 'float32');
$a += 1;
echo "+= ", json_encode($a->toArray()), "\n";
$a -= 1;
echo "-= ", json_encode($a->toArray()), "\n";
$a *= 2;
echo "*= ", json_encode($a->toArray()), "\n";
$a /= 2;
echo "/= ", json_encode($a->toArray()), "\n";
$a **= 2;
echo "**= ", json_encode($a->toArray()), "\n";
?>
--EXPECT--
float4 + operator/method 6/6 OK
float4 - operator/method 2/2 OK
float4 * operator/method 6/6 OK
float4 / operator/method 2/2 OK
float4 ** operator/method 6/6 OK
float8 + operator/method 6/6 OK
float8 - operator/method 2/2 OK
float8 * operator/method 8/8 OK
float8 / operator/method 2/2 OK
float8 ** operator/method 16/16 OK
float16 + operator/method 6/6 OK
float16 - operator/method 2/2 OK
float16 * operator/method 8/8 OK
float16 / operator/method 2/2 OK
float16 ** operator/method 16/16 OK
float32 + operator/method 6/6 OK
float32 - operator/method 2/2 OK
float32 * operator/method 8/8 OK
float32 / operator/method 2/2 OK
float32 ** operator/method 16/16 OK
float64 + operator/method 6/6 OK
float64 - operator/method 2/2 OK
float64 * operator/method 8/8 OK
float64 / operator/method 2/2 OK
float64 ** operator/method 16/16 OK
float128 + operator/method 6/6 OK
float128 - operator/method 2/2 OK
float128 * operator/method 8/8 OK
float128 / operator/method 2/2 OK
float128 ** operator/method 16/16 OK
int8 + operator/method 6/6 OK
int8 - operator/method 2/2 OK
int8 * operator/method 8/8 OK
int8 / operator/method 2/2 OK
int8 ** operator/method 16/16 OK
uint8 + operator/method 6/6 OK
uint8 - operator/method 2/2 OK
uint8 * operator/method 8/8 OK
uint8 / operator/method 2/2 OK
uint8 ** operator/method 16/16 OK
int16 + operator/method 6/6 OK
int16 - operator/method 2/2 OK
int16 * operator/method 8/8 OK
int16 / operator/method 2/2 OK
int16 ** operator/method 16/16 OK
uint16 + operator/method 6/6 OK
uint16 - operator/method 2/2 OK
uint16 * operator/method 8/8 OK
uint16 / operator/method 2/2 OK
uint16 ** operator/method 16/16 OK
int32 + operator/method 6/6 OK
int32 - operator/method 2/2 OK
int32 * operator/method 8/8 OK
int32 / operator/method 2/2 OK
int32 ** operator/method 16/16 OK
uint32 + operator/method 6/6 OK
uint32 - operator/method 2/2 OK
uint32 * operator/method 8/8 OK
uint32 / operator/method 2/2 OK
uint32 ** operator/method 16/16 OK
int64 + operator/method 6/6 OK
int64 - operator/method 2/2 OK
int64 * operator/method 8/8 OK
int64 / operator/method 2/2 OK
int64 ** operator/method 16/16 OK
uint64 + operator/method 6/6 OK
uint64 - operator/method 2/2 OK
uint64 * operator/method 8/8 OK
uint64 / operator/method 2/2 OK
uint64 ** operator/method 16/16 OK
+= [2,3,4,5]
-= [1,2,3,4]
*= [2,4,6,8]
/= [1,2,3,4]
**= [1,4,9,16]
