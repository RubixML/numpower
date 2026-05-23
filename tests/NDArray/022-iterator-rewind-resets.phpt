--TEST--
NDArray Iterator: rewind() resets the cursor; current()/key() are stable
--FILE--
<?php
/* Iterator semantics that PHP foreach relies on:
   1. current() and key() do NOT advance the cursor -- calling them N times
      in a row returns the same value (otherwise foreach could skip).
   2. rewind() resets from any mid-traversal state, including from past-end.
   3. A second foreach over the same array yields the same sequence. */

$a = new NDArray([10, 20, 30, 40, 50], 'int32');

/* Stability: identical results across repeated calls. */
$a->rewind();
echo "stable key/current x3:\n";
for ($i = 0; $i < 3; $i++) {
    echo "  key=", $a->key(), " current=", $a->current(), "\n";
}

/* Advance, then current()/key() must reflect the new cursor stably. */
$a->next();
$a->next();
echo "after 2x next():\n";
for ($i = 0; $i < 3; $i++) {
    echo "  key=", $a->key(), " current=", $a->current(), "\n";
}

/* Rewind from mid-traversal. */
$a->rewind();
echo "after rewind from mid:\n";
echo "  key=", $a->key(), " current=", $a->current(), "\n";

/* Walk to the end. */
for ($i = 0; $i < 5; $i++) $a->next();
echo "past end: valid=", $a->valid() ? '1' : '0',
     " current=", $a->current() === null ? 'null' : (string)$a->current(), "\n";

/* Rewind from past-end. */
$a->rewind();
echo "after rewind from past-end:\n";
echo "  key=", $a->key(), " current=", $a->current(),
     " valid=", $a->valid() ? '1' : '0', "\n";

/* Two consecutive foreach passes yield the same sequence. */
$seq1 = [];
foreach ($a as $k => $v) { $seq1[] = "$k:$v"; }
$seq2 = [];
foreach ($a as $k => $v) { $seq2[] = "$k:$v"; }
echo "foreach equal: ", $seq1 === $seq2 ? '1' : '0', "\n";
echo "foreach seq: ", implode(',', $seq1), "\n";
?>
--EXPECT--
stable key/current x3:
  key=0 current=10
  key=0 current=10
  key=0 current=10
after 2x next():
  key=2 current=30
  key=2 current=30
  key=2 current=30
after rewind from mid:
  key=0 current=10
past end: valid=0 current=null
after rewind from past-end:
  key=0 current=10 valid=1
foreach equal: 1
foreach seq: 0:10,1:20,2:30,3:40,4:50
