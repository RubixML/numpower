--TEST--
slice() error handling on both NDArray::slice and NumPower::slice: bounds, dim count, step=0, 0-D source
--FILE--
<?php
/* slice() must reject invalid inputs cleanly with descriptive messages,
   never silently produce a garbage result or segfault. Use the static
   NumPower::slice() form: instance slice now mutates $this, which would
   leave the test source in an unpredictable state between sub-tests. */

function tryslice(callable $f, string $tag) {
    try {
        $r = $f();
        echo "$tag: NO ERROR (got " . (is_object($r) ? "NDArray" : var_export($r, true)) . ")\n";
    } catch (\Throwable $e) {
        echo "$tag: ", $e->getMessage(), "\n";
    }
}

/* 0-D source */
$scalar = new NDArray(3.14);
tryslice(fn() => NumPower::slice($scalar, 0), "0-D source");

/* Too many indices */
$v = NumPower::arange(5.0);
tryslice(fn() => NumPower::slice($v, 0, 0), "too many indices on 1D");

$m = NumPower::array([[1.0, 2.0], [3.0, 4.0]]);
tryslice(fn() => NumPower::slice($m, 0, 0, 0), "too many indices on 2D");

/* Step = 0 */
tryslice(fn() => NumPower::slice($v, [0, 5, 0]), "step=0");

/* Single-int out-of-bounds (positive and negative) */
tryslice(fn() => NumPower::slice($v, 5),  "single int 5 of 5");
tryslice(fn() => NumPower::slice($v, -6), "single int -6 of 5");
tryslice(fn() => NumPower::slice($v, 100), "single int 100");

/* Negative single int normalised in range (-1, -5) must succeed */
echo "v[-1]=", NumPower::slice($v, -1), "\n";
echo "v[-5]=", NumPower::slice($v, -5), "\n";

/* No-index static call — ZEND_PARSE_PARAMETERS requires ≥2 args */
tryslice(fn() => NumPower::slice($v), "no idx args");

/* No args on instance — ArgumentCountError */
tryslice(fn() => $v->slice(), "instance no args");

/* Slice spec longer than 3 */
tryslice(fn() => NumPower::slice($v, [0, 5, 1, 1]), "4-element slice spec");

/* Slice spec on second axis where caller passes [] for axis 0 */
$ok = NumPower::slice($m, [], 0)->toArray();
echo "slice(m, [], 0) = ", json_encode($ok), "\n";

/* Range that produces zero elements is allowed, returns shape [0] */
$z = NumPower::slice($v, [3, 3]);
echo "slice(v, [3,3]) shape=", json_encode($z->shape()), " size=", $z->size(), "\n";

/* Reverse range with positive step yields empty (Slice_GetIndices semantic) */
$z2 = NumPower::slice($v, [4, 1]);
echo "slice(v, [4,1]) shape=", json_encode($z2->shape()), "\n";

/* Out-of-bounds range gets clamped, not errored */
$clamped = NumPower::slice($v, [0, 999]);
echo "slice(v, [0,999]) = ", $clamped, "\n";

/* The source survived the entire test (proves NumPower::slice() is pure). */
echo "v after all slices: shape=", json_encode($v->shape()), " val=$v\n";
echo "m after all slices: shape=", json_encode($m->shape()), "\n";
?>
--EXPECTF--
0-D source: slice is not defined for a 0-d array
too many indices on 1D: too many indices for array.
too many indices on 2D: too many indices for array.
step=0: slice step cannot be zero
single int 5 of 5: index 5 is out of bounds for axis 0 with size 5
single int -6 of 5: index -6 is out of bounds for axis 0 with size 5
single int 100: index 100 is out of bounds for axis 0 with size 5
v[-1]=4
v[-5]=0
no idx args: %s
instance no args: %s
4-element slice spec: slice spec on axis 0 has 4 values; expected 0-3
slice(m, [], 0) = [1,3]
slice(v, [3,3]) shape=[0] size=0
slice(v, [4,1]) shape=[0]
slice(v, [0,999]) = [0, 1, 2, 3, 4]
v after all slices: shape=[5] val=[0, 1, 2, 3, 4]
m after all slices: shape=[2,2]
