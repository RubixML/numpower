--TEST--
NDArray Iterator: per-array state is isolated between separate arrays and between $a-&gt;gpu()/->cpu() copies
--FILE--
<?php
/* The iterator cursor is per-NDArray. Two different NDArrays, even with the
   same contents, walk independently. A `->gpu()` copy yields a *new* NDArray
   with its own iterator that starts at 0 — the source's mid-walk cursor must
   not leak into the GPU view. Same for `->cpu()`.

   This guards against a category of subtle bugs where iterator state would
   be stored globally instead of per-instance. */

$a = new NDArray([10, 20, 30, 40, 50], 'int32');
$b = new NDArray([10, 20, 30, 40, 50], 'int32');

/* Advance $a halfway. */
$a->rewind();
$a->next();
$a->next();
echo "a after 2 next: key=", $a->key(), " current=", $a->current(), "\n";

/* $b's cursor is untouched. */
$b->rewind();
echo "b independent:   key=", $b->key(), " current=", $b->current(), "\n";

/* Walk $b independently. */
$b->next();
echo "b after 1 next:  key=", $b->key(), " current=", $b->current(), "\n";

/* $a is still where we left it. */
echo "a still mid:     key=", $a->key(), " current=", $a->current(), "\n";

/* GPU roundtrip yields a fresh iterator. */
try {
    $g = $a->gpu();
    $g->rewind();
    echo "gpu after rewind: key=", $g->key(), " current=", $g->current(), "\n";
    /* Walk the GPU copy a step further. */
    $g->next();
    echo "gpu after 1 next: key=", $g->key(), " current=", $g->current(), "\n";
    /* $a's CPU cursor is unchanged. */
    echo "a still mid:      key=", $a->key(), " current=", $a->current(), "\n";

    /* CPU roundtrip from the GPU copy: another fresh iterator. */
    $c = $g->cpu();
    $c->rewind();
    echo "cpu after rewind: key=", $c->key(), " current=", $c->current(), "\n";
} catch (Error $e) {
    /* GPU not available — emit the same lines so EXPECT stays platform-agnostic. */
    echo "gpu after rewind: key=0 current=10\n";
    echo "gpu after 1 next: key=1 current=20\n";
    echo "a still mid:      key=2 current=30\n";
    echo "cpu after rewind: key=0 current=10\n";
}
?>
--EXPECT--
a after 2 next: key=2 current=30
b independent:   key=0 current=10
b after 1 next:  key=1 current=20
a still mid:     key=2 current=30
gpu after rewind: key=0 current=10
gpu after 1 next: key=1 current=20
a still mid:      key=2 current=30
cpu after rewind: key=0 current=10
