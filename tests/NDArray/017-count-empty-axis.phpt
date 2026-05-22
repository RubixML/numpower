--TEST--
NDArray::count() returns 0 for arrays with an empty leading axis (shape [0, ...])
--FILE--
<?php
/* An NDArray with shape[0] == 0 is a legitimate empty array along axis 0
   (e.g. an empty batch). count() must return 0, foreach must do 0 iterations,
   and the result must match across CPU and GPU. */

function assert_empty_axis(NDArray $a, string $tag): void {
    $c = $a->count();
    $b = count($a);
    $iters = 0;
    foreach ($a as $_) {
        $iters++;
        if ($iters > 100) {
            echo "FAIL $tag: foreach did not terminate\n";
            return;
        }
    }
    if ($c !== 0 || $b !== 0 || $iters !== 0) {
        echo "FAIL $tag: count()=$c builtin=$b foreach=$iters\n";
        return;
    }
    echo "OK $tag\n";
}

/* Empty leading axis at various ranks. */
assert_empty_axis(NumPower::zeros([0, 3]),        "[0,3]");
assert_empty_axis(NumPower::zeros([0, 5, 7]),     "[0,5,7]");
assert_empty_axis(NumPower::zeros([0, 2, 3, 4]),  "[0,2,3,4]");

/* Empty along inner axes still leaves shape[0] intact. */
$inner_empty = NumPower::zeros([5, 0]);
echo "shape[0]=5,shape[1]=0: count()=" . $inner_empty->count() . " size=" . $inner_empty->size() . PHP_EOL;
?>
--EXPECT--
OK [0,3]
OK [0,5,7]
OK [0,2,3,4]
shape[0]=5,shape[1]=0: count()=5 size=0
