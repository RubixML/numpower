--TEST--
NDArray::cast() boundaries: every dtype pair, 0-D, empty and N-D arrays
--FILE--
<?php
/* 1. Every source dtype to every target dtype (14 x 14). Values 1..3 are
      exactly representable in all of them, including float4/float8.
      float128 and uint64 come back as PHP strings to keep full precision. */
$types = ['float4', 'float8', 'float16', 'float32', 'float64', 'float128',
          'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'];
$bad = [];
foreach ($types as $from) {
    foreach ($types as $to) {
        $a = new NDArray([1, 2, 3], $from);
        $a->cast($to);
        $want = match (true) {
            in_array($to, ['float128', 'uint64'], true) => ['1', '2', '3'],
            $to[0] === 'f'                              => [1.0, 2.0, 3.0],
            default                                     => [1, 2, 3],
        };
        if ($a->dataType() !== $to || $a->toArray() !== $want) {
            $bad[] = "$from->$to";
        }
    }
}
echo "pairs: ", count($types) ** 2, " bad: ", json_encode($bad), "\n";

/* 2. 0-D array stays an NDArray with the new dtype */
$z = new NDArray(2.7, 'float32');
$z->cast('int32');
echo "0-D: ", get_class($z), ' ', $z->dataType(), ' ', json_encode($z->toArray()), "\n";

/* 3. Empty array */
$e = NumPower::zeros([0]);
$e->cast('int8');
echo "empty: ", $e->dataType(), ' ', json_encode($e->shape()), ' ', json_encode($e->toArray()), "\n";

/* 4. N-D shape and values preserved */
$m = NumPower::ones([2, 3, 4]);
$m->cast('int16');
$flat = array_merge(...array_merge(...$m->toArray()));
echo "3-D: ", $m->dataType(), ' ', json_encode($m->shape()), ' all ones: ',
     ($flat === array_fill(0, 24, 1) ? 'yes' : 'no'), "\n";

/* 5. Truncation toward zero for float -> int */
$t = new NDArray([1.9, -1.9, 0.5, -0.5], 'float64');
$t->cast('int32');
echo "truncate: ", json_encode($t->toArray()), "\n";

/* 6. Repeated casts back and forth do not leak */
$x = NumPower::ones([1000]);
$x->cast('int16');
$m0 = memory_get_usage();
for ($i = 0; $i < 10000; $i++) {
    $x->cast($i % 2 ? 'int16' : 'float32');
}
echo "leak: ", (memory_get_usage() - $m0 < 64 * 1024 ? 'no' : 'yes'), "\n";
?>
--EXPECT--
pairs: 196 bad: []
0-D: NDArray int32 2
empty: int8 [0] []
3-D: int16 [2,3,4] all ones: yes
truncate: [1,-1,0,0]
leak: no
