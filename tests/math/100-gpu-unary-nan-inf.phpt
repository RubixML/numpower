--TEST--
NumPower unary ops propagate NaN / Inf correctly on GPU for float16 / float32 / float64
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); }
catch (\Error $e) { die("skip " . $e->getMessage()); }
?>
--FILE--
<?php
/* IEEE-754 NaN / Inf propagation on the GPU kernel path. The float16
   kernels in particular round-trip through `__float2half` /
   `__half2float`, and `__float2half(NaN)` must yield a representable
   NaN (any qNaN payload). */

foreach (['float32', 'float64', 'float16'] as $dt) {
    $tol = ($dt === 'float16') ? 1e-2 : 1e-6;

    $a   = NumPower::array([NAN, INF, -INF, 0.0, -1.0], $dt)->gpu();
    $abs = NumPower::abs($a)->cpu()->toArray();
    echo "$dt abs[NaN]    is NaN: ",   is_nan($abs[0]) ? "yes" : "no", "\n";
    echo "$dt abs[+Inf]   is +Inf: ",  ($abs[1] === INF) ? "yes" : "no", "\n";
    echo "$dt abs[-Inf]   is +Inf: ",  ($abs[2] === INF) ? "yes" : "no", "\n";
    echo "$dt abs[0]      = ", $abs[3], "\n";
    echo "$dt abs[-1]     = ", $abs[4], "\n";

    $sgn = NumPower::sign($a)->cpu()->toArray();
    echo "$dt sign[+Inf]  = ", $sgn[1], "\n";
    echo "$dt sign[-Inf]  = ", $sgn[2], "\n";
    echo "$dt sign[0]     = ", $sgn[3], "\n";
    echo "$dt sign[-1]    = ", $sgn[4], "\n";

    /* sqrt(-1) = NaN on GPU float kernels. */
    $sq = NumPower::sqrt(NumPower::array([-1.0, 0.0, 4.0], $dt)->gpu())->cpu()->toArray();
    echo "$dt sqrt[-1]   is NaN: ", is_nan($sq[0]) ? "yes" : "no", "\n";
    echo "$dt sqrt[0]    = ", $sq[1], "\n";
    echo "$dt sqrt[4]    = ", $sq[2], "\n";

    /* reciprocal(0) = +Inf. */
    $rec = NumPower::reciprocal(NumPower::array([0.0, 1.0, -2.0], $dt)->gpu())->cpu()->toArray();
    echo "$dt recip[0]   is Inf: ", is_infinite($rec[0]) ? "yes" : "no", "\n";
    echo "$dt recip[1]   = ", $rec[1], "\n";

    /* sinc continuity at 0 on GPU. */
    $sc = NumPower::sinc(NumPower::array([0.0, 1.0], $dt)->gpu())->cpu()->toArray();
    echo "$dt sinc[0]    = ", $sc[0], "\n";
    if (!(abs($sc[1]) < $tol)) echo "FAIL $dt sinc(1) = ", $sc[1], " not near 0\n";
}

/* int32 → float64 promotion on GPU still NaN-safe for sqrt(-1). */
$rs = NumPower::sqrt(NumPower::array([-1, 0, 9], 'int32')->gpu())->cpu()->toArray();
echo "int32 promoted sqrt[-1] is NaN: ", is_nan($rs[0]) ? "yes" : "no", "\n";
echo "int32 promoted sqrt[0]  = ", $rs[1], "\n";
echo "int32 promoted sqrt[9]  = ", $rs[2], "\n";

echo "DONE\n";
?>
--EXPECT--
float32 abs[NaN]    is NaN: yes
float32 abs[+Inf]   is +Inf: yes
float32 abs[-Inf]   is +Inf: yes
float32 abs[0]      = 0
float32 abs[-1]     = 1
float32 sign[+Inf]  = 1
float32 sign[-Inf]  = -1
float32 sign[0]     = 0
float32 sign[-1]    = -1
float32 sqrt[-1]   is NaN: yes
float32 sqrt[0]    = 0
float32 sqrt[4]    = 2
float32 recip[0]   is Inf: yes
float32 recip[1]   = 1
float32 sinc[0]    = 1
float64 abs[NaN]    is NaN: yes
float64 abs[+Inf]   is +Inf: yes
float64 abs[-Inf]   is +Inf: yes
float64 abs[0]      = 0
float64 abs[-1]     = 1
float64 sign[+Inf]  = 1
float64 sign[-Inf]  = -1
float64 sign[0]     = 0
float64 sign[-1]    = -1
float64 sqrt[-1]   is NaN: yes
float64 sqrt[0]    = 0
float64 sqrt[4]    = 2
float64 recip[0]   is Inf: yes
float64 recip[1]   = 1
float64 sinc[0]    = 1
float16 abs[NaN]    is NaN: yes
float16 abs[+Inf]   is +Inf: yes
float16 abs[-Inf]   is +Inf: yes
float16 abs[0]      = 0
float16 abs[-1]     = 1
float16 sign[+Inf]  = 1
float16 sign[-Inf]  = -1
float16 sign[0]     = 0
float16 sign[-1]    = -1
float16 sqrt[-1]   is NaN: yes
float16 sqrt[0]    = 0
float16 sqrt[4]    = 2
float16 recip[0]   is Inf: yes
float16 recip[1]   = 1
float16 sinc[0]    = 1
int32 promoted sqrt[-1] is NaN: yes
int32 promoted sqrt[0]  = 0
int32 promoted sqrt[9]  = 3
DONE
