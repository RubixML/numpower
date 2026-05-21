--TEST--
unserialize() accepts the legacy (pre-fix) serialize format
--FILE--
<?php
/* Before the dtype-aware __serialize fix, NDArray returned a plain numeric
   array from __serialize() and ended up serialised as:
     O:7:"NDArray":N:{i:0;d:...;i:1;d:...;...}
   Old data persisted in caches / databases must still unserialize after the
   upgrade. We construct the legacy format manually and verify it. */

/* 1-D legacy payload — three floats */
$legacy_1d = 'O:7:"NDArray":3:{i:0;d:1.5;i:1;d:2.5;i:2;d:3.5;}';
$a = unserialize($legacy_1d);
echo "1D legacy: ", $a->toArray() === [1.5, 2.5, 3.5] ? "OK" : "BAD", "\n";

/* 2-D legacy payload — 2x2 */
$legacy_2d = 'O:7:"NDArray":2:{i:0;a:2:{i:0;d:1.5;i:1;d:2.5;}i:1;a:2:{i:0;d:3.5;i:1;d:4.5;}}';
$b = unserialize($legacy_2d);
echo "2D legacy: ", $b->toArray() === [[1.5, 2.5], [3.5, 4.5]] ? "OK" : "BAD", "\n";

/* Sanity: the legacy fallback produces a float32 array (the old default).
   Verify by checking element types are floats. */
$first = $a->toArray()[0];
echo "legacy dtype default: ", (is_float($first) ? "float (OK)" : "BAD"), "\n";
?>
--EXPECT--
1D legacy: OK
2D legacy: OK
legacy dtype default: float (OK)
