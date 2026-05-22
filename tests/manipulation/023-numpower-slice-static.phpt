--TEST--
NumPower::slice() static-method contract: pure, accepts NDArray|array|scalar input, never mutates source
--FILE--
<?php
/* The static `NumPower::slice($source, ...$indices)` is a pure-function
   alternative to the instance method. It produces an independent NDArray
   (or dtype-correct scalar for 0-D), and the source is never modified.
   It also accepts a wider input — anything `ZVAL_TO_NDARRAY` resolves —
   so PHP arrays and bare scalars work as the source. */

/* ---- (1) Source NDArray is untouched after every slice spec form ---- */
$src = new NDArray([[10, 20, 30, 40], [50, 60, 70, 80]], 'int32');
$snapshot = $src->toArray();

NumPower::slice($src, 0);
NumPower::slice($src, [], -1);
NumPower::slice($src, [0, 2], [1, 4]);
NumPower::slice($src, 1, 2);              /* 0-D result */
NumPower::slice($src, [-1, -3, -1]);      /* negative step */
echo "source after 5 slices: shape=", json_encode($src->shape()),
     " values_intact=", ($src->toArray() === $snapshot ? "yes" : "NO"), "\n";

/* ---- (2) PHP-array source is accepted ---- */
$out = NumPower::slice([[1, 2, 3], [4, 5, 6]], 0);
echo "array source: ",
     ($out->toArray() === [1.0, 2.0, 3.0] ? "OK" : "BAD got=" . json_encode($out->toArray())),
     "\n";

/* ---- (3) Result is independent — modifying it does not bleed into src ---- */
$src2 = new NDArray([[1, 2, 3], [4, 5, 6]], 'int32');
$res  = NumPower::slice($src2, 0);
$res->fill(99);
echo "result-mutation independence: ",
     ($src2->toArray() === [[1, 2, 3], [4, 5, 6]] ? "OK" : "BAD"), "\n";
echo "filled result: ", json_encode($res->toArray()), "\n";

/* ---- (4) Same arg forms as the instance method ---- */
$m = NumPower::array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
echo "static row0:   ", NumPower::slice($m, 0),        "\n";
echo "static col-1:  ", NumPower::slice($m, [], -1),   "\n";
echo "static sub:    ", NumPower::slice($m, 0, [0, 2]),"\n";

/* ---- (5) 0-D scalar return (dtype-aware) ---- */
$i64 = new NDArray([100, 200, 300], 'int64');
$s = NumPower::slice($i64, 1);
echo "scalar from int64: ", var_export($s, true), " (type=", gettype($s), ")\n";

$str = new NDArray(['1.5', '2.5'], 'float128');
$s2 = NumPower::slice($str, 1);
echo "scalar from fp128 type=", gettype($s2), " val=", $s2, "\n";

/* ---- (6) Errors propagate ---- */
try { NumPower::slice($src, 99); echo "no throw\n"; }
catch (\Throwable $e) { echo "static OOB throws: ", $e->getMessage(), "\n"; }

try { NumPower::slice($src, 0, 0, 0, 0); echo "no throw\n"; }
catch (\Throwable $e) { echo "static too-many-axes throws: ", $e->getMessage(), "\n"; }

/* ---- (7) After all the above, source is *still* unchanged ---- */
echo "final src intact: ",
     ($src->toArray() === $snapshot ? "OK" : "BAD"), "\n";
?>
--EXPECT--
source after 5 slices: shape=[2,4] values_intact=yes
array source: OK
result-mutation independence: OK
filled result: [99,99,99]
static row0:   [1, 2, 3]
static col-1:  [3, 6]
static sub:    [1, 2]
scalar from int64: 200 (type=integer)
scalar from fp128 type=string val=2.5
static OOB throws: index 99 is out of bounds for axis 0 with size 2
static too-many-axes throws: too many indices for array.
final src intact: OK
