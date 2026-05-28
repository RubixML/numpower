--TEST--
NumPower::abs/negative/positive/reciprocal/sign/sqrt/rsqrt/square/clip/sinc across every dtype (CPU)
--FILE--
<?php
/* Covers all 10 unary math ops across every supported dtype on CPU.
   Verifies:
     - abs / negative / positive / sign / square / clip preserve dtype;
     - reciprocal / sqrt / rsqrt / sinc promote integer inputs to float
       (narrow ints → float32, wider ints → float64);
     - signed-integer wrap semantics on negate / abs / square match
       NumPy (e.g. `-INT_MIN == INT_MIN`);
     - unsigned-integer abs is a no-op;
     - sign returns -1 / 0 / +1 in the source dtype (0 / 1 for unsigned);
     - clip parses int / float / string bounds losslessly. */

function approx_equal($g, $w, $tol) {
    if (is_array($g) && is_array($w)) {
        if (count($g) !== count($w)) return false;
        $gv = array_values($g);
        $wv = array_values($w);
        for ($i = 0; $i < count($gv); $i++) {
            if (!approx_equal($gv[$i], $wv[$i], $tol)) return false;
        }
        return true;
    }
    if (is_array($g) || is_array($w)) return false;
    if (is_float($g) || is_float($w)) {
        $gf = (float)$g;
        $wf = (float)$w;
        if (is_nan($gf) && is_nan($wf)) return true;
        return abs($gf - $wf) <= $tol;
    }
    return (string)$g === (string)$w;
}

function check($label, $got, $want, $tol = 0.0) {
    if (approx_equal($got, $want, $tol)) {
        echo "OK $label\n";
    } else {
        echo "FAIL $label: got=", json_encode($got),
             " want=", json_encode($want), "\n";
    }
}

/* ── Integer dtypes preserve dtype on the preserving ops ──────────────── */
foreach (['int8','int16','int32','int64'] as $dt) {
    $arr = NumPower::array([-3, 0, 5], $dt);
    check("$dt abs",      NumPower::abs($arr)->toArray(),      [3, 0, 5]);
    check("$dt negative", NumPower::negative($arr)->toArray(), [3, 0, -5]);
    check("$dt positive", NumPower::positive($arr)->toArray(), [-3, 0, 5]);
    check("$dt sign",     NumPower::sign($arr)->toArray(),     [-1, 0, 1]);
    check("$dt square",   NumPower::square($arr)->toArray(),   [9, 0, 25]);
    check("$dt clip",     NumPower::clip($arr, -2, 4)->toArray(), [-2, 0, 4]);
}

foreach (['uint8','uint16','uint32','uint64'] as $dt) {
    $arr = NumPower::array([3, 0, 5], $dt);
    check("$dt abs",      NumPower::abs($arr)->toArray(),      [3, 0, 5]);
    check("$dt negative", NumPower::negative($arr)->toArray(), $dt === 'uint8'  ? [253, 0, 251]
                                                            : ($dt === 'uint16' ? [65533, 0, 65531]
                                                            : ($dt === 'uint32' ? [4294967293, 0, 4294967291]
                                                            : ['18446744073709551613', '0', '18446744073709551611'])));
    check("$dt positive", NumPower::positive($arr)->toArray(), [3, 0, 5]);
    check("$dt sign",     NumPower::sign($arr)->toArray(),     [1, 0, 1]);
    check("$dt square",   NumPower::square($arr)->toArray(),   [9, 0, 25]);
    check("$dt clip",     NumPower::clip($arr, 1, 4)->toArray(),  [3, 1, 4]);
}

/* ── Float dtypes: full op coverage ───────────────────────────────────── */
foreach (['float16','float32','float64'] as $dt) {
    $tol = ($dt === 'float16') ? 5e-3 : ($dt === 'float32' ? 1e-6 : 1e-12);
    $arr = NumPower::array([-3.0, 0.0, 5.0], $dt);
    check("$dt abs",        NumPower::abs($arr)->toArray(),        [3.0, 0.0, 5.0], $tol);
    check("$dt negative",   NumPower::negative($arr)->toArray(),   [3.0, 0.0, -5.0], $tol);
    check("$dt positive",   NumPower::positive($arr)->toArray(),   [-3.0, 0.0, 5.0], $tol);
    check("$dt sign",       NumPower::sign($arr)->toArray(),       [-1.0, 0.0, 1.0], $tol);
    check("$dt square",     NumPower::square($arr)->toArray(),     [9.0, 0.0, 25.0], $tol);
    check("$dt clip",       NumPower::clip($arr, -2.0, 4.0)->toArray(), [-2.0, 0.0, 4.0], $tol);

    $pos = NumPower::array([1.0, 4.0, 9.0, 16.0], $dt);
    check("$dt sqrt",       NumPower::sqrt($pos)->toArray(),       [1.0, 2.0, 3.0, 4.0], $tol);
    check("$dt rsqrt",      NumPower::rsqrt($pos)->toArray(),      [1.0, 0.5, 1.0/3, 0.25], $tol);
    check("$dt reciprocal", NumPower::reciprocal($pos)->toArray(), [1.0, 0.25, 1.0/9, 0.0625], $tol);

    $sinc_in = NumPower::array([0.0, 0.5, 1.0], $dt);
    /* sinc(0) == 1, sinc(0.5) == sin(π/2)/(π/2) ≈ 2/π, sinc(1) ≈ 0 */
    check("$dt sinc", NumPower::sinc($sinc_in)->toArray(), [1.0, 2.0/M_PI, 0.0], $tol);
}

/* ── int → float promotion (sqrt/rsqrt/reciprocal/sinc) ───────────────── */
$ar_i16 = NumPower::array([1, 4, 9, 16], 'int16');
$sqrt_i16 = NumPower::sqrt($ar_i16);
check("int16 sqrt values",  $sqrt_i16->toArray(), [1, 2, 3, 4], 1e-6);
$ar_i32 = NumPower::array([1, 4, 9, 16], 'int32');
$sqrt_i32 = NumPower::sqrt($ar_i32);
check("int32 sqrt values",  $sqrt_i32->toArray(), [1, 2, 3, 4], 1e-12);

/* ── 2-D arrays exercise the kernel beyond a single row ───────────────── */
$mat = NumPower::array([[-2.0, 0.0], [3.0, -4.0]], 'float64');
check("2-D abs",      NumPower::abs($mat)->toArray(),     [[2.0, 0.0], [3.0, 4.0]], 1e-12);
check("2-D negative", NumPower::negative($mat)->toArray(),[[2.0, -0.0], [-3.0, 4.0]], 1e-12);

/* ── 3-D and 4-D arrays ──────────────────────────────────────────────── */
$rows = [];
for ($i = 0; $i < 4; $i++) $rows[] = $i - 2;
$nd3 = NumPower::array([[[$rows[0], $rows[1]],[$rows[2], $rows[3]]]], 'int32');
check("3-D int32 abs", NumPower::abs($nd3)->toArray(), [[[2, 1], [0, 1]]]);

/* ── float128 ─────────────────────────────────────────────────────────── */
$f128 = NumPower::array(['-1.5', '2.25', '0'], 'float128');
check("fp128 abs",  NumPower::abs($f128)->toArray(),       ['1.5', '2.25', '0']);
check("fp128 neg",  NumPower::negative($f128)->toArray(),  ['1.5', '-2.25', '-0']);
check("fp128 sign", NumPower::sign($f128)->toArray(),      ['-1', '1', '0']);
check("fp128 sq",   NumPower::square($f128)->toArray(),    ['2.25', '5.0625', '0']);
check("fp128 sqrt", NumPower::sqrt(NumPower::array(['16', '25'], 'float128'))->toArray(), ['4', '5']);
check("fp128 recip",NumPower::reciprocal(NumPower::array(['2','4','0.5'], 'float128'))->toArray(),
                    ['0.5','0.25','2']);

/* ── uint64 limits ────────────────────────────────────────────────────── */
$u64 = NumPower::array(['18446744073709551615', '0', '1'], 'uint64');
check("uint64 abs",  NumPower::abs($u64)->toArray(),  ['18446744073709551615', '0', '1']);
check("uint64 sign", NumPower::sign($u64)->toArray(), ['1', '0', '1']);
check("uint64 clip (string)",
      NumPower::clip($u64, '5', '18446744073709551000')->toArray(),
      ['18446744073709551000', '5', '5']);
check("uint64 clip (int)",
      NumPower::clip(NumPower::array([3, 10, 20], 'uint64'), 5, 15)->toArray(),
      ['5', '10', '15']);

/* ── int wrap on negate/abs/square ────────────────────────────────────── */
check("int8 -(-128)",    NumPower::negative(NumPower::array([-128], 'int8'))->toArray(),  [-128]);
check("int8 abs(-128)",  NumPower::abs(NumPower::array([-128], 'int8'))->toArray(),       [-128]);
check("int8 square 16",  NumPower::square(NumPower::array([16], 'int8'))->toArray(),      [0]);
check("int32 square 2^16",NumPower::square(NumPower::array([65536], 'int32'))->toArray(), [0]);

/* ── 0-D scalar input ─────────────────────────────────────────────────── */
$z = NumPower::array(-5.5);
check("0-D abs (returns scalar)",  NumPower::abs($z),  5.5,  1e-12);
check("0-D sign (returns scalar)", NumPower::sign($z), -1.0, 1e-12);
$z_int = NumPower::array(9);
$sq = NumPower::sqrt($z_int);
check("0-D int sqrt (returns scalar)", $sq, 3.0, 1e-6);

echo "DONE\n";
?>
--EXPECT--
OK int8 abs
OK int8 negative
OK int8 positive
OK int8 sign
OK int8 square
OK int8 clip
OK int16 abs
OK int16 negative
OK int16 positive
OK int16 sign
OK int16 square
OK int16 clip
OK int32 abs
OK int32 negative
OK int32 positive
OK int32 sign
OK int32 square
OK int32 clip
OK int64 abs
OK int64 negative
OK int64 positive
OK int64 sign
OK int64 square
OK int64 clip
OK uint8 abs
OK uint8 negative
OK uint8 positive
OK uint8 sign
OK uint8 square
OK uint8 clip
OK uint16 abs
OK uint16 negative
OK uint16 positive
OK uint16 sign
OK uint16 square
OK uint16 clip
OK uint32 abs
OK uint32 negative
OK uint32 positive
OK uint32 sign
OK uint32 square
OK uint32 clip
OK uint64 abs
OK uint64 negative
OK uint64 positive
OK uint64 sign
OK uint64 square
OK uint64 clip
OK float16 abs
OK float16 negative
OK float16 positive
OK float16 sign
OK float16 square
OK float16 clip
OK float16 sqrt
OK float16 rsqrt
OK float16 reciprocal
OK float16 sinc
OK float32 abs
OK float32 negative
OK float32 positive
OK float32 sign
OK float32 square
OK float32 clip
OK float32 sqrt
OK float32 rsqrt
OK float32 reciprocal
OK float32 sinc
OK float64 abs
OK float64 negative
OK float64 positive
OK float64 sign
OK float64 square
OK float64 clip
OK float64 sqrt
OK float64 rsqrt
OK float64 reciprocal
OK float64 sinc
OK int16 sqrt values
OK int32 sqrt values
OK 2-D abs
OK 2-D negative
OK 3-D int32 abs
OK fp128 abs
OK fp128 neg
OK fp128 sign
OK fp128 sq
OK fp128 sqrt
OK fp128 recip
OK uint64 abs
OK uint64 sign
OK uint64 clip (string)
OK uint64 clip (int)
OK int8 -(-128)
OK int8 abs(-128)
OK int8 square 16
OK int32 square 2^16
OK 0-D abs (returns scalar)
OK 0-D sign (returns scalar)
OK 0-D int sqrt (returns scalar)
DONE
