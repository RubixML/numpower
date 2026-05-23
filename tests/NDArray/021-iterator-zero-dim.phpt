--TEST--
NDArray Iterator: 0-D source -- valid() is always false, current() is null
--FILE--
<?php
/* A 0-D NDArray has no axis 0 to enumerate. The Iterator contract for this
   case must be:
     - valid()   -> false unconditionally
     - current() -> null (matches PHP's standard "past end" convention)
     - key()     -> 0 (the never-advanced initial cursor)
     - foreach   -> zero iterations

   Reading shape[0] of a 0-D array would dereference past the dimensions
   buffer; the guard inside iterator_is_done() must short-circuit that. */

$dtypes = ['float32', 'float64', 'float128', 'int8', 'int64', 'uint64'];

foreach ($dtypes as $t) {
    $scalar = new NDArray(7, $t);

    /* Explicit driver: must terminate immediately. */
    $scalar->rewind();
    $v = $scalar->valid();
    $c = $scalar->current();
    $k = $scalar->key();
    echo "$t explicit: valid=", $v ? '1' : '0',
         " current=", $c === null ? 'null' : 'NOTNULL',
         " key=$k\n";

    /* foreach: must run zero iterations. */
    $iters = 0;
    foreach ($scalar as $key => $val) { $iters++; }
    echo "$t foreach: iters=$iters\n";
}
?>
--EXPECT--
float32 explicit: valid=0 current=null key=0
float32 foreach: iters=0
float64 explicit: valid=0 current=null key=0
float64 foreach: iters=0
float128 explicit: valid=0 current=null key=0
float128 foreach: iters=0
int8 explicit: valid=0 current=null key=0
int8 foreach: iters=0
int64 explicit: valid=0 current=null key=0
int64 foreach: iters=0
uint64 explicit: valid=0 current=null key=0
uint64 foreach: iters=0
