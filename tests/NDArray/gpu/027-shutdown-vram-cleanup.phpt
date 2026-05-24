--TEST--
NDArray buffer_free() releases surviving VRAM slots during RSHUTDOWN
--SKIPIF--
<?php try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); } ?>
--ENV--
NDARRAY_VCHECK=1
--FILE--
<?php
/* Regression for a request-shutdown ordering leak.
 *
 * PHP's php_request_shutdown calls zend_deactivate_modules() — which runs
 * our PHP_RSHUTDOWN_FUNCTION (buffer_free + vmemcheck) — *before* it calls
 * zend_deactivate(), which is where the symbol table is cleaned up and
 * free_obj handlers (ndarray_destructor) finally fire. Any NDArray still
 * referenced by a global PHP variable when RSHUTDOWN starts would slip
 * through: buffer_free() previously released only the buffer array, never
 * walking the slots, so surviving NDArrays' VRAM (or CPU buffers) leaked.
 *
 * The leak surfaced only at refcount > 1 because a single global pointing
 * at a GPU NDArray happened to be released by the engine before our
 * buffer_free in this PHP build, while two globals aliasing one object
 * deferred destruction past it. Reproduce both patterns plus a more
 * involved chain to make sure the walk in buffer_free() catches every
 * surviving slot.
 *
 * The probe is the EXPECT body itself — any leftover allocation triggers
 * `VRAM MEMORY LEAK: leaked N array(s)` on stderr/stdout from vmemcheck()
 * and breaks the EXPECT match. */

/* Pattern A: aliased global. */
$a = (new NDArray([1.0, 2.0, 3.0, 4.0], 'float32'))->gpu();
$b = $a;

/* Pattern B: idempotent gpu() — same PHP object via ZVAL_COPY. */
$c = (new NDArray([1.0, 2.0], 'float64'))->gpu();
$c2 = $c->gpu();
$c3 = $c2->gpu()->gpu();

/* Pattern C: a chain that mixes a real CPU->GPU transfer and idempotent
   calls per dtype, leaving everything live until RSHUTDOWN. */
$keep = [];
foreach (['float32', 'float64', 'float128', 'int8', 'uint64'] as $dt) {
    $k = (new NDArray([1, 2, 3, 4], $dt))->gpu();
    $keep[$dt]      = $k;
    $keep[$dt.'_b'] = $k;            // aliased
    $keep[$dt.'_c'] = $k->gpu();     // idempotent => same object as $k
}

/* Pattern D: VRAM-direct zeros() left alive. */
$z = NumPower::zeros([8, 8], 'float64', NUMPOWER_CUDA);
$z2 = $z;

echo "live_at_shutdown_ok\n";
?>
--EXPECT--
live_at_shutdown_ok
