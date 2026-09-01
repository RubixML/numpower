--TEST--
NumPower::append / concatenate / columnStack on float64 (byte-correct element copy)
--FILE--
<?php
use NumPower as nd;

/* Regression: raw_array_assign_array used to copy sizeof(float) bytes per
   element (4) while strides were elsize-based (8 for float64). The result
   buffer kept the zero-filled upper 4 bytes of every element, so a float64
   output read back as a subnormal near zero — wildly wrong. The fix copies
   NDArray_ELSIZE(dst) bytes per element.

   Values are chosen with non-trivial upper 4 bytes of the IEEE-754 double so
   ANY loss in the high half of the bytes is caught. */

/* ---- append (axis=-1 → ConcatenateFlat) ---- */
$a1 = nd::array([0.111111111, 0.222222222], 'float64');
$b1 = nd::array([0.333333333, 0.444444444], 'float64');
$expected_append = [0.111111111, 0.222222222, 0.333333333, 0.444444444];
$got_append = nd::append($a1, $b1)->toArray();
$ok_append = (count($got_append) === 4) &&
             abs($got_append[0] - $expected_append[0]) < 1e-15 &&
             abs($got_append[1] - $expected_append[1]) < 1e-15 &&
             abs($got_append[2] - $expected_append[2]) < 1e-15 &&
             abs($got_append[3] - $expected_append[3]) < 1e-15;
echo "append float64: ", $ok_append ? "OK" : "BAD got=" . json_encode($got_append), "\n";

/* ---- concatenate axis=0 (vstack) ---- */
$a2 = nd::array([
    [0.123456789,  0.987654321],
    [0.314159265,  0.271828182],
], 'float64');
$b2 = nd::array([
    [0.500000001,  0.600000002],
    [0.700000003,  0.800000004],
], 'float64');
$expected_v = [
    [0.123456789, 0.987654321],
    [0.314159265, 0.271828182],
    [0.500000001, 0.600000002],
    [0.700000003, 0.800000004],
];
$got_v = nd::concatenate([$a2, $b2], 0)->toArray();
$ok_v = (count($got_v) === 4);
if ($ok_v) {
    for ($i = 0; $i < 4; $i++) {
        for ($j = 0; $j < 2; $j++) {
            if (abs($got_v[$i][$j] - $expected_v[$i][$j]) > 1e-15) {
                $ok_v = false;
            }
        }
    }
}
echo "concatenate axis=0: ", $ok_v ? "OK" : "BAD got=" . json_encode($got_v), "\n";

/* ---- concatenate axis=1 (hstack) ---- */
$expected_h = [
    [0.123456789, 0.987654321, 0.500000001, 0.600000002],
    [0.314159265, 0.271828182, 0.700000003, 0.800000004],
];
$got_h = nd::concatenate([$a2, $b2], 1)->toArray();
$ok_h = (count($got_h) === 2 && count($got_h[0]) === 4);
if ($ok_h) {
    for ($i = 0; $i < 2; $i++) {
        for ($j = 0; $j < 4; $j++) {
            if (abs($got_h[$i][$j] - $expected_h[$i][$j]) > 1e-15) {
                $ok_h = false;
            }
        }
    }
}
echo "concatenate axis=1: ", $ok_h ? "OK" : "BAD got=" . json_encode($got_h), "\n";

/* ---- columnStack (each array transposed then concat axis=1 → 2x4) ---- */
$expected_cs = [
    [0.123456789, 0.314159265, 0.500000001, 0.700000003],
    [0.987654321, 0.271828182, 0.600000002, 0.800000004],
];
$got_cs = nd::columnStack([$a2, $b2])->toArray();
$ok_cs = (count($got_cs) === 2 && count($got_cs[0]) === 4);
if ($ok_cs) {
    for ($i = 0; $i < 2; $i++) {
        for ($j = 0; $j < 4; $j++) {
            if (abs($got_cs[$i][$j] - $expected_cs[$i][$j]) > 1e-15) {
                $ok_cs = false;
            }
        }
    }
}
echo "columnStack: ", $ok_cs ? "OK" : "BAD got=" . json_encode($got_cs), "\n";
?>
--EXPECT--
append float64: OK
concatenate axis=0: OK
concatenate axis=1: OK
columnStack: OK
