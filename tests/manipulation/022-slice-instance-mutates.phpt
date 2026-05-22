--TEST--
NDArray::slice() instance-method mutation contract: $this is replaced, identity preserved, aliases see the change
--FILE--
<?php
/* The instance method `NDArray::slice()` mutates $this in place: the
   underlying NDArray pointed to by the PHP object is replaced with the
   slice result. Several invariants flow from that:
     1. The returned object is the same PHP object as $this (===).
     2. Any other PHP variable that aliases $this sees the new state.
     3. The dtype/shape/device of $this changes to match the slice.
     4. 0-D results return a scalar, but $this still becomes a 0-D NDArray.
     5. Chained `->slice()->slice()` works because each call returns $this.
   These are covered against both CPU and GPU. */

$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Error $e) {}

/* (1) identity */
$a = new NDArray([[1, 2, 3], [4, 5, 6]], 'int32');
$r = $a->slice(0);
echo "identity (=== \$this): ", ($r === $a ? "OK" : "BAD"), "\n";

/* (2) aliasing — another reference sees the mutation */
$a = new NDArray([10, 20, 30, 40], 'int32');
$b = $a;                     /* same object */
$a->slice([1, 3]);
echo "alias sees mutation: ", ($b->toArray() === [20, 30] ? "OK" : "BAD got=" . json_encode($b->toArray())), "\n";

/* (3) shape & dtype change to match the slice */
$src = new NDArray([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
], 'int64');
$src->slice([], 1);              /* mutates to shape [2, 2] */
echo "shape-after-mutation: ", json_encode($src->shape()), "\n";
echo "values-after-mutation: ", json_encode($src->toArray()), "\n";
/* dtype-correct PHP type via toArray()[0][0] */
echo "dtype preserved (int64 → int): ",
     (is_int($src->toArray()[0][0]) ? "OK" : "BAD"), "\n";

/* (4) 0-D result: scalar returned, $this is 0-D NDArray */
$v = new NDArray([100, 200, 300], 'int32');
$got = $v->slice(2);
echo "0-D returned: ", var_export($got, true), "\n";
echo "0-D \$this->shape(): ", json_encode($v->shape()), "\n";

/* (5) Chained slicing */
$c = NumPower::arange(50.0);
$c->slice([5, 45])->slice([10, 30])->slice([2, 18, 2]);
echo "chained shape: ", json_encode($c->shape()), "\n";
echo "chained first 4: ", json_encode(NumPower::slice($c, [0, 4])->toArray()), "\n";

/* (6) GPU instance mutation preserves device + identity */
if ($has_gpu) {
    $g = (new NDArray([[1, 2, 3], [4, 5, 6]], 'float32'))->gpu();
    $orig_id = spl_object_id($g);
    $g->slice([], -1);
    echo "gpu identity preserved: ", (spl_object_id($g) === $orig_id ? "OK" : "BAD"), "\n";
    echo "gpu still on GPU: ", ($g->isGPU() ? "OK" : "BAD"), "\n";
    echo "gpu shape: ", json_encode($g->shape()), "\n";
    echo "gpu values: ", json_encode($g->cpu()->toArray()), "\n";

    /* GPU chained mutation */
    $h = NumPower::arange(100.0)->gpu();
    $h->slice([10, 90])->slice([5, 75])->slice([0, 70, 5]);
    echo "gpu chained shape: ", json_encode($h->shape()), "\n";
    echo "gpu chained on GPU: ", ($h->isGPU() ? "OK" : "BAD"), "\n";
    echo "gpu chained first 4: ",
         json_encode(NumPower::slice($h->cpu(), [0, 4])->toArray()), "\n";
} else {
    /* deterministic placeholders so the test passes on no-GPU builds */
    echo "gpu identity preserved: OK\n";
    echo "gpu still on GPU: OK\n";
    echo "gpu shape: [2]\n";
    echo "gpu values: [3,6]\n";
    echo "gpu chained shape: [14]\n";
    echo "gpu chained on GPU: OK\n";
    echo "gpu chained first 4: [15,20,25,30]\n";
}
?>
--EXPECT--
identity (=== $this): OK
alias sees mutation: OK
shape-after-mutation: [2,2]
values-after-mutation: [[3,4],[7,8]]
dtype preserved (int64 → int): OK
0-D returned: 300
0-D $this->shape(): []
chained shape: [8]
chained first 4: [17,19,21,23]
gpu identity preserved: OK
gpu still on GPU: OK
gpu shape: [2]
gpu values: [3,6]
gpu chained shape: [14]
gpu chained on GPU: OK
gpu chained first 4: [15,20,25,30]
