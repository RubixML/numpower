--TEST--
Cross-device arithmetic throws consistent device mismatch errors; same-device GPU ops work
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
$a  = new NDArray([[1,2],[3,4]], 'float32');
$b  = new NDArray([[3,4],[5,6]], 'float32');
$ag = $a->gpu();
$bg = $b->gpu();

// CPU + GPU must throw
try {
    $a + $bg;
    echo "FAIL: no error for cpu+gpu\n";
} catch (Error $e) {
    echo $e->getMessage() . "\n";
}

// GPU + CPU must throw
try {
    $ag + $b;
    echo "FAIL: no error for gpu+cpu\n";
} catch (Error $e) {
    echo $e->getMessage() . "\n";
}

// GPU + GPU same dtype (float32) should succeed
$r = ($ag + $bg)->cpu();
print_r($r->toArray());
?>
--EXPECT--
Device mismatch, both NDArray MUST be in the same device.
Device mismatch, both NDArray MUST be in the same device.
Array
(
    [0] => Array
        (
            [0] => 4
            [1] => 6
        )

    [1] => Array
        (
            [0] => 8
            [1] => 10
        )

)
