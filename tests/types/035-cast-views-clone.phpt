--TEST--
NDArray::cast() with slices and clones: only the cast object changes
--FILE--
<?php
/* 1. Casting a slice changes only the slice */
$p = new NDArray([[1.5, 2.5], [3.5, 4.5]], 'float32');
$s = $p[1];
$s->cast('int32');
echo "slice:  ", $s->dataType(), ' ', json_encode($s->toArray()), "\n";
echo "parent: ", $p->dataType(), ' ', json_encode($p->toArray()), "\n";

/* 2. Casting the parent keeps an existing slice valid and unchanged */
$p = new NDArray([[1.5, 2.5], [3.5, 4.5]], 'float32');
$s = $p[0];
$p->cast('int8');
echo "parent: ", $p->dataType(), ' ', json_encode($p->toArray()), "\n";
echo "slice:  ", $s->dataType(), ' ', json_encode($s->toArray()), "\n";
unset($p);
echo "slice after parent freed: ", json_encode($s->toArray()), "\n";

/* 3. A clone is independent */
$a = new NDArray([1.5, 2.5], 'float32');
$c = clone $a;
$c->cast('int8');
echo "orig:  ", $a->dataType(), ' ', json_encode($a->toArray()), "\n";
echo "clone: ", $c->dataType(), ' ', json_encode($c->toArray()), "\n";

/* 4. Arithmetic works on a cast array */
$x = new NDArray([1.5, 2.5], 'float32');
$x->cast('int32');
echo "add: ", json_encode(NumPower::add($x, new NDArray([1, 1], 'int32'))->toArray()), "\n";
?>
--EXPECT--
slice:  int32 [3,4]
parent: float32 [[1.5,2.5],[3.5,4.5]]
parent: int8 [[1,2],[3,4]]
slice:  float32 [1.5,2.5]
slice after parent freed: [1.5,2.5]
orig:  float32 [1.5,2.5]
clone: int8 [1,2]
add: [2,3]
