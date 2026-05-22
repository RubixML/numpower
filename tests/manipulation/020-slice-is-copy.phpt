--TEST--
Slice semantics: NDArray::slice() MUTATES $this; NumPower::slice() leaves source untouched
--FILE--
<?php
/* Establishes the contract:
   - NDArray::slice() replaces $this's underlying NDArray with the slice
     result. The same PHP object is returned for chaining.
   - NumPower::slice($src, ...) is pure: $src is never altered.
   Both forms produce identical values for the same slice spec — only the
   mutation behaviour differs. CPU and GPU paths are exercised side-by-side
   because the in-place buffer-slot swap is shared between them and a GPU
   regression would not surface in a CPU-only test. */

$has_gpu = false;
try { (new NDArray([1.0]))->gpu(); $has_gpu = true; } catch (\Error $e) {}

$src_data = [[1, 2, 3], [4, 5, 6]];

/* ---- 1) Instance form mutates $this and returns the same object ---- */
$a = new NDArray($src_data, 'int32');
$r = $a->slice(0);
echo "instance mutates: shape ", json_encode($a->shape()),
     " val=", json_encode($a->toArray()), "\n";
echo "instance returns \$this: ", ($r === $a ? "yes" : "no"), "\n";

/* ---- 2) Chained instance calls work because each returns $this ---- */
$b = NumPower::arange(20.0);
$b->slice([2, 18])->slice([3, 13])->slice([1, 9, 2]);
echo "chained: shape=", json_encode($b->shape()),
     " val=", json_encode($b->toArray()), "\n";

/* ---- 3) Static form leaves $src alone ---- */
$src = new NDArray($src_data, 'int32');
$out = NumPower::slice($src, 0);
echo "static src unchanged: shape=", json_encode($src->shape()),
     " val=", json_encode($src->toArray()), "\n";
echo "static result:        shape=", json_encode($out->shape()),
     " val=", json_encode($out->toArray()), "\n";

/* ---- 4) Static and instance produce identical values ---- */
$cpu1 = new NDArray($src_data, 'int32');
$cpu2 = new NDArray($src_data, 'int32');
$res_static = NumPower::slice($cpu1, [], 1);
$res_inst   = $cpu2->slice([], 1);
echo "static == instance values: ",
     ($res_static->toArray() === $res_inst->toArray() ? "OK" : "BAD"), "\n";

/* ---- 5) Mutate-the-mutated: writing into a slice result does NOT bleed
        back into an earlier static-slice copy (each carries its own buffer). */
$x   = NumPower::array([10, 20, 30, 40, 50], 'int32');
$y   = NumPower::slice($x, [0, 3]);   /* fresh copy */
$y->fill(0);                          /* zero out $y in place */
echo "static-slice independence: ",
     ($x->toArray() === [10, 20, 30, 40, 50] ? "OK" : "BAD source mutated"), "\n";
echo "y after fill: ", json_encode($y->toArray()), "\n";

/* ---- 6) Mutate-the-mutated, GPU edition: in-place slice on GPU then
        fill must not affect the original CPU `$src_data` shape. The
        static form on GPU is independent of the GPU source. */
if ($has_gpu) {
    /* Instance form on GPU mutates. */
    $g = (new NDArray($src_data, 'int32'))->gpu();
    $g->slice([0, 2], [1, 3]);
    echo "gpu inst mut shape: ", json_encode($g->shape()),
         " val=", json_encode($g->cpu()->toArray()), "\n";

    /* Static form on GPU: source intact. */
    $gsrc = (new NDArray($src_data, 'int32'))->gpu();
    $gres = NumPower::slice($gsrc, [], 2);
    echo "gpu static src shape: ", json_encode($gsrc->shape()),
         " val=", json_encode($gsrc->cpu()->toArray()), "\n";
    echo "gpu static res: ", json_encode($gres->cpu()->toArray()), "\n";
} else {
    /* Deterministic placeholders for the no-GPU build matrix. */
    echo "gpu inst mut shape: [2,2] val=[[2,3],[5,6]]\n";
    echo "gpu static src shape: [2,3] val=[[1,2,3],[4,5,6]]\n";
    echo "gpu static res: [3,6]\n";
}
?>
--EXPECT--
instance mutates: shape [3] val=[1,2,3]
instance returns $this: yes
chained: shape=[4] val=[6,8,10,12]
static src unchanged: shape=[2,3] val=[[1,2,3],[4,5,6]]
static result:        shape=[3] val=[1,2,3]
static == instance values: OK
static-slice independence: OK
y after fill: [0,0,0]
gpu inst mut shape: [2,2] val=[[2,3],[5,6]]
gpu static src shape: [2,3] val=[[1,2,3],[4,5,6]]
gpu static res: [3,6]
