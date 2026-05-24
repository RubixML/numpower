--TEST--
NumPower::fromImage() rejects invalid arguments with clear error messages
--SKIPIF--
<?php if (!extension_loaded('gd')) die('skip GD extension not loaded'); ?>
--FILE--
<?php
/* Validation errors must be catchable \Error instances — never silent
   bogus NDArrays or segfaults. Covers: non-GdImage input, unknown
   dtype, out-of-range device id. */

try {
    NumPower::fromImage(42);
    echo "non-image: BAD (no throw)\n";
} catch (\Error $e) {
    echo "non-image: OK\n";
}

try {
    NumPower::fromImage([1, 2, 3]);
    echo "array-input: BAD (no throw)\n";
} catch (\Error $e) {
    echo "array-input: OK\n";
}

try {
    NumPower::fromImage(new stdClass());
    echo "stdClass: BAD (no throw)\n";
} catch (\Error $e) {
    echo "stdClass: OK\n";
}

$img = imagecreatetruecolor(2, 2);

try {
    NumPower::fromImage($img, true, "float256");
    echo "bad-dtype: BAD (no throw)\n";
} catch (\Error $e) {
    echo "bad-dtype: OK\n";
}

try {
    NumPower::fromImage($img, true, "uint8", 99);
    echo "bad-device: BAD (no throw)\n";
} catch (\Error $e) {
    echo "bad-device: OK\n";
}

try {
    NumPower::fromImage($img, true, "uint8", -1);
    echo "negative-device: BAD (no throw)\n";
} catch (\Error $e) {
    echo "negative-device: OK\n";
}
?>
--EXPECT--
non-image: OK
array-input: OK
stdClass: OK
bad-dtype: OK
bad-device: OK
negative-device: OK
