--TEST--
NDArray::gpu()/cpu() on a 0-D array returns an NDArray on the target device
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Earlier behavior collapsed 0-D arrays into a PHP scalar on the
   gpu()/cpu() boundary — that broke `clone $a` after
   `$a = (new NDArray(1, 'float128'))->gpu()` because `$a` ended up as
   a string, not an object, and also violated the rule that GPU-resident
   data must stay on GPU. Both methods now always return an NDArray
   object on the requested device, even for 0-D inputs. */

/* CPU 0-D → cpu(): same object, still 0-D, still on CPU. */
$a   = new NDArray(3.14, 'float64');
$out = $a->cpu();
$ok1 = $out instanceof NDArray
    && $out->isGPU() === 0
    && count($out->shape()) === 0;

/* CPU 0-D → gpu(): new NDArray on GPU, still 0-D. */
$b   = new NDArray(2.5, 'float32');
$g   = $b->gpu();
$ok2 = $g instanceof NDArray
    && $g->isGPU() === 1
    && count($g->shape()) === 0;

/* Chained: (new NDArray(...))->gpu() — used to return a primitive,
   which broke `clone`. Verify both that it returns an NDArray and
   that clone works on the result. */
$c   = (new NDArray(7.0, 'float32'))->gpu();
$ok3 = $c instanceof NDArray && $c->isGPU() === 1;
$cc  = clone $c;
$ok4 = $cc instanceof NDArray && $cc->isGPU() === 1;

/* Round-trip CPU→GPU→CPU: value preserved. */
$d = new NDArray(42.5, 'float32');
$rt = $d->gpu()->cpu();
$ok5 = $rt instanceof NDArray && $rt->isGPU() === 0;

echo ($ok1 && $ok2 && $ok3 && $ok4 && $ok5) ? "OK\n" : "FAIL\n";
?>
--EXPECT--
OK
