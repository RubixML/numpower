--TEST--
NDArray offsetGet / offsetSet throw on non-integer-coercible offsets (NaN, Inf, bool, null, oversized double)
--FILE--
<?php
/* offsetExists silently returns false for non-integer offsets (test 030);
   offsetGet and offsetSet must throw a clean Zend Error instead — never
   crash, never trip the wrong code path. Of particular interest:

     - NaN / +Inf / -Inf: a naive (long)double cast invokes UB and would
       typically land on LONG_MIN, which the bounds check would then
       diagnose as "negative index" (misleading).
     - 1e30 / -1e30: overflow signed long — same UB risk.
     - true / false: NDArray treats booleans as non-integer per the strict
       typed-array contract; PHP arrays would accept them, but NDArray
       indices must be explicit integers.

   The exact message is part of the contract; we assert it matches. */

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
    'string'        => 'abc',
    'array'         => [0],
];

foreach ($cases as $name => $off) {
    $get_err = 'NONE';
    try { $x = $a[$off]; } catch (\Throwable $e) { $get_err = $e->getMessage(); }
    $set_err = 'NONE';
    try { $a[$off] = 1.0; } catch (\Throwable $e) { $set_err = $e->getMessage(); }
    echo "$name: get='$get_err' set='$set_err'\n";
}

/* The source array must be untouched after every failed offsetSet. */
echo "shape=", json_encode($a->shape()), " values=", json_encode($a->toArray()), "\n";

/* Negative index throws a specific (different) message — keep the contract. */
$err = 'NONE';
try { $x = $a[-1]; } catch (\Throwable $e) { $err = $e->getMessage(); }
echo "neg: $err\n";

/* Out-of-range throws yet a different message. */
$err = 'NONE';
try { $x = $a[99]; } catch (\Throwable $e) { $err = $e->getMessage(); }
echo "oor: $err\n";
?>
--EXPECT--
null: get='Invalid offset' set='Invalid offset'
true: get='Invalid offset' set='Invalid offset'
false: get='Invalid offset' set='Invalid offset'
NaN: get='Invalid offset' set='Invalid offset'
+Inf: get='Invalid offset' set='Invalid offset'
-Inf: get='Invalid offset' set='Invalid offset'
oversize-pos: get='Invalid offset' set='Invalid offset'
oversize-neg: get='Invalid offset' set='Invalid offset'
string: get='Invalid offset' set='Invalid offset'
array: get='Invalid offset' set='Invalid offset'
shape=[3] values=[1,2,3]
neg: Negative indexes are not implemented.
oor: Index out of bounds
