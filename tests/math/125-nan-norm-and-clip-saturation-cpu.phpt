--TEST--
NaN-sign normalization (all fp dtypes) + integer clip-bound saturation/inf/nan (all int dtypes), CPU
--FILE--
<?php
/* Covers two branch behaviours that previously had no direct test:

   1. NaN-sign normalization (src/debug.c + src/ndarray_types.c): every
      floating dtype must render NaN as the unsigned literal "nan" — never
      glibc's "-nan" — across __toString, matching PyTorch / Python repr.
      A sign-bit-set NaN (produced by `negative(NaN)`) must still print
      "nan" while the stored bit pattern keeps the flipped sign.

   2. Integer clip-bound saturation (src/ndmath/arithmetics.c,
      unary_parse_typed_scalar): a finite out-of-range string bound must
      saturate to the dtype range (PyTorch clamp), and an inf/nan bound
      must act as "no bound" (−inf → MIN, +inf → MAX; nan → the no-op
      extreme for its side) rather than collapsing to 0 via strtoll. */

/* ── 1. NaN-sign normalization, every fp dtype ───────────────────────── */
echo "=== NaN normalization (CPU) ===\n";
foreach (['float16', 'float32', 'float64', 'float128'] as $dt) {
    /* log(-1) = NaN in the middle element; the outer elements verify the
       rest of the row prints normally. */
    $r = NumPower::log(NumPower::array([1.0, -1.0, 4.0], $dt));
    $s = (string)$r;
    $ok = (strpos($s, 'nan') !== false) && (strpos($s, '-nan') === false)
        && (strpos($s, 'NAN') === false);
    echo "$dt: " . ($ok ? "OK" : "FAIL ($s)") . "\n";
}

echo "=== negative(NaN) prints 'nan' but flips the stored sign bit ===\n";
foreach (['float32', 'float64', 'float128'] as $dt) {
    $neg = NumPower::negative(NumPower::array([NAN], $dt));
    $pos = NumPower::positive(NumPower::array([NAN], $dt));
    $sn = (string)$neg;
    $ok = (strpos($sn, '-nan') === false) && (strpos($sn, 'nan') !== false);
    echo "$dt neg=" . ($ok ? "nan-ok" : "FAIL($sn)")
        . " pos=" . ((string)$pos === '[nan]' ? "nan-ok" : "FAIL") . "\n";
}

/* ── 2. integer clip-bound saturation, finite OOB ────────────────────── */
echo "\n=== Finite out-of-range clip bounds saturate (PyTorch clamp) ===\n";
function show($label, $arr) { echo "$label: [" . implode(",", $arr) . "]\n"; }

show("int8  [-100,0,100,127,-128] clip(-50,50)",
     NumPower::clip(NumPower::array([-100,0,100,127,-128], "int8"), "-50", "50")->toArray());
show("int8  [-100,0,100,127,-128] clip(-300,300)",
     NumPower::clip(NumPower::array([-100,0,100,127,-128], "int8"), "-300", "300")->toArray());
show("int16 [-30000,0,30000] clip(-10000,10000)",
     NumPower::clip(NumPower::array([-30000,0,30000], "int16"), "-10000", "10000")->toArray());
show("int16 [-30000,0,30000] clip(-40000,40000)",
     NumPower::clip(NumPower::array([-30000,0,30000], "int16"), "-40000", "40000")->toArray());
show("int32 [-2e9,0,2e9] clip(-1e9,1e9)",
     NumPower::clip(NumPower::array([-2000000000,0,2000000000], "int32"), "-1000000000", "1000000000")->toArray());
show("uint8 [0,100,200,255] clip(-50,150)",
     NumPower::clip(NumPower::array([0,100,200,255], "uint8"), "-50", "150")->toArray());
show("uint16 [0,50000,65535] clip(-5,70000)",
     NumPower::clip(NumPower::array([0,50000,65535], "uint16"), "-5", "70000")->toArray());
show("uint32 [0,2e9,4e9] clip(-5,3e9)",
     NumPower::clip(NumPower::array([0,2000000000,4000000000], "uint32"), "-5", "3000000000")->toArray());

/* ── inf / nan clip bounds on integer dtypes ─────────────────────────── */
echo "\n=== inf / nan integer clip bounds act as 'no bound' ===\n";
show("int8 [-50,0,50] clip(-inf,inf)",
     NumPower::clip(NumPower::array([-50,0,50], "int8"), "-inf", "inf")->toArray());
show("int8 [-50,0,50] clip(-inf,10)",
     NumPower::clip(NumPower::array([-50,0,50], "int8"), "-inf", "10")->toArray());
show("int8 [-50,0,50] clip(nan,10)",
     NumPower::clip(NumPower::array([-50,0,50], "int8"), "nan", "10")->toArray());
show("int8 [-50,0,50] clip(-10,nan)",
     NumPower::clip(NumPower::array([-50,0,50], "int8"), "-10", "nan")->toArray());
show("uint8 [10,200,50] clip(-inf,inf)",
     NumPower::clip(NumPower::array([10,200,50], "uint8"), "-inf", "inf")->toArray());

/* ── malformed inf/nan bound is now rejected (strict validator) ──────── */
echo "\n=== Malformed inf/nan clip bound rejected ===\n";
foreach (["infinityX", "nanZ", "infX"] as $bad) {
    try {
        NumPower::clip(NumPower::array([1.0], "float64"), "0.0", $bad);
        echo "$bad: NOT REJECTED\n";
    } catch (\Throwable $e) {
        echo "$bad: rejected\n";
    }
}

echo "DONE\n";
?>
--EXPECT--
=== NaN normalization (CPU) ===
float16: OK
float32: OK
float64: OK
float128: OK
=== negative(NaN) prints 'nan' but flips the stored sign bit ===
float32 neg=nan-ok pos=nan-ok
float64 neg=nan-ok pos=nan-ok
float128 neg=nan-ok pos=nan-ok

=== Finite out-of-range clip bounds saturate (PyTorch clamp) ===
int8  [-100,0,100,127,-128] clip(-50,50): [-50,0,50,50,-50]
int8  [-100,0,100,127,-128] clip(-300,300): [-100,0,100,127,-128]
int16 [-30000,0,30000] clip(-10000,10000): [-10000,0,10000]
int16 [-30000,0,30000] clip(-40000,40000): [-30000,0,30000]
int32 [-2e9,0,2e9] clip(-1e9,1e9): [-1000000000,0,1000000000]
uint8 [0,100,200,255] clip(-50,150): [0,100,150,150]
uint16 [0,50000,65535] clip(-5,70000): [0,50000,65535]
uint32 [0,2e9,4e9] clip(-5,3e9): [0,2000000000,3000000000]

=== inf / nan integer clip bounds act as 'no bound' ===
int8 [-50,0,50] clip(-inf,inf): [-50,0,50]
int8 [-50,0,50] clip(-inf,10): [-50,0,10]
int8 [-50,0,50] clip(nan,10): [-50,0,10]
int8 [-50,0,50] clip(-10,nan): [-10,0,50]
uint8 [10,200,50] clip(-inf,inf): [10,200,50]

=== Malformed inf/nan clip bound rejected ===
infinityX: rejected
nanZ: rejected
infX: rejected
DONE
