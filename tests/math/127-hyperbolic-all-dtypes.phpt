--TEST--
NumPower hyperbolic family (sinh/cosh/tanh/arcsinh/arccosh/arctanh): all dtypes, boundaries, domains, string intake (CPU)
--FILE--
<?php
/* Pure-CPU coverage for the six hyperbolic ops, which ride the typed unary
   dispatcher (NDArray_TypedUnaryOp) shared with the trig / exp-log family:
   every floating dtype is preserved, integer inputs widen to float
   (narrow → float32, 32/64-bit → float64), and a numeric-string scalar is
   accepted with the dtype inferred from the literal (float128 for a
   decimal, uint64/int64 for an integer).

   Reference for float16/32/64 is PHP's own libm (sinh/cosh/… built-ins) —
   the same library PyTorch/NumPy use, so matching it == matching PyTorch
   for those dtypes. float128 is spot-checked against decimal-computed
   constants to 28 significant digits (safe on both the libquadmath ~34-digit
   and the double-double ~32-digit backends) and cross-checked with the
   identities cosh²−sinh²=1 and arcsinh(sinh(x))=x.

   Silent on success: only failures print, then a fixed summary, so the
   strict --EXPECT-- turns any value mismatch into a CI failure. No GPU
   calls here, so it runs on every runner; CPU↔GPU parity lives in 128. */

$FAILS = 0;
function ok($cond, $label) {
    global $FAILS;
    if (!$cond) { echo "FAIL: $label\n"; $FAILS++; }
}
/* Tolerant scalar compare: NaN==NaN, sign-aware inf, relative-or-absolute. */
function near($g, $w, $tol) {
    $gf = (float)$g; $wf = (float)$w;
    if (is_nan($gf) || is_nan($wf)) return is_nan($gf) && is_nan($wf);
    if (is_infinite($gf) || is_infinite($wf))
        return is_infinite($gf) && is_infinite($wf) && (($gf < 0) === ($wf < 0));
    if ($wf == 0.0) return abs($gf) <= $tol;
    return abs($gf - $wf) <= max($tol, abs($wf) * $tol);
}
/* Leading agreeing significant digits between two decimal strings. */
function sig($a, $b) {
    $na = ltrim(str_replace(['-', '.'], '', $a), '0');
    $nb = ltrim(str_replace(['-', '.'], '', $b), '0');
    if ($na === '' && $nb === '') return 99;          /* both zero */
    $n = min(strlen($na), strlen($nb));
    for ($i = 0; $i < $n; $i++) if ($na[$i] !== $nb[$i]) return $i;
    return $n;
}
$OPS = ['sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh'];
/* NumPower op name → PHP libm built-in (arc* → a*). */
$LIBM = ['sinh' => 'sinh', 'cosh' => 'cosh', 'tanh' => 'tanh',
         'arcsinh' => 'asinh', 'arccosh' => 'acosh', 'arctanh' => 'atanh'];

/* ── float16 / float32 / float64 vs PHP libm ───────────────────────────── */
/* Domain-aware inputs: arccosh needs x ≥ 1, arctanh needs |x| < 1. */
$dom = [
    'sinh'    => [0.0, 0.0001, 0.5, 1.0, 2.5, -0.5, -3.0],
    'cosh'    => [0.0, 0.0001, 0.5, 1.0, 2.5, -0.5, -3.0],
    'tanh'    => [0.0, 0.0001, 0.5, 1.0, 5.0, -0.5, -2.0],
    'arcsinh' => [0.0, 0.0001, 0.5, 1.0, 10.0, -0.5, -4.0],
    'arccosh' => [1.0, 1.5, 2.0, 10.0, 100.0],
    'arctanh' => [0.0, 0.0001, 0.5, 0.9, -0.5, -0.95],
];
$tol = ['float16' => 2e-3, 'float32' => 2e-6, 'float64' => 1e-12];
foreach (['float16', 'float32', 'float64'] as $dt) {
    foreach ($OPS as $op) {
        foreach ($dom[$op] as $x) {
            $got = NumPower::$op(new NDArray([$x], $dt))[0];
            $exp = $LIBM[$op]((float)$x);     /* PHP libm reference */
            ok(near($got, $exp, $tol[$dt]), "$op($x) $dt vs libm (got=$got exp=$exp)");
        }
    }
}

/* ── integer dtypes widen to float, values via libm ────────────────────── */
/* narrow ints (int8..uint16) → float32; 32/64-bit ints → float64. */
$narrow = ['int8', 'uint8', 'int16', 'uint16'];
$wide   = ['int32', 'uint32', 'int64', 'uint64'];
foreach (array_merge($narrow, $wide) as $dt) {
    $isWide = in_array($dt, $wide, true);
    $r = NumPower::sinh(new NDArray([1, 2], $dt));
    ok($r->__serialize()['dtype'] === ($isWide ? 'float64' : 'float32'),
       "sinh($dt) result dtype");
    ok(near($r[0], sinh(1.0), 2e-6) && near($r[1], sinh(2.0), 2e-6),
       "sinh($dt) values widened via libm");
}
/* arccosh on an integer ≥ 1; arctanh on integer 0 (only in-domain int). */
ok(near(NumPower::arccosh(new NDArray([2], 'int32'))[0], acosh(2.0), 1e-12),
   "arccosh(int 2)");
ok((float)NumPower::arctanh(new NDArray([0], 'int16'))[0] === 0.0,
   "arctanh(int 0) == 0");

/* ── float128 spot checks (28 sig digits) + identities ─────────────────── */
$REF = [
    ['sinh',    '0.5', '0.521095305493747361622425626411491559106'],
    ['sinh',    '-2',  '-3.626860407847018767668213982801261704886'],
    ['cosh',    '0.5', '1.127625965206380785226225161402672012548'],
    ['cosh',    '3',   '10.06766199577776584195393603511588983681'],
    ['tanh',    '0.5', '0.4621171572600097585023184836436725487303'],
    ['tanh',    '-1',  '-0.7615941559557648881194582826047935904127'],
    ['arcsinh', '0.5', '0.4812118250596034474977589134243684231350'],
    ['arcsinh', '-10', '-2.998222950297969738846595537596453476507'],
    ['arccosh', '1.5', '0.9624236501192068949955178268487368462703'],
    ['arccosh', '2',   '1.316957896924816708625046347307968444027'],
    ['arctanh', '0.5', '0.5493061443340548456976226184612628523235'],
    ['arctanh', '-0.9','-1.472219489583220230004513715943926768618'],
];
foreach ($REF as [$op, $x, $ref]) {
    $got = (string) NumPower::$op(new NDArray([$x], 'float128'))[0];
    ok(sig($ref, $got) >= 28, "fp128 $op($x) = $got (ref $ref)");
}
/* Identities at fp128 precision via NumPower's own fp128 arithmetic.
   cosh(x)² − sinh(x)² == 1 to ≳30 digits. */
foreach (['0.5', '1.5', '3'] as $x) {
    $a  = new NDArray([$x], 'float128');
    $ch = NumPower::cosh($a);
    $sh = NumPower::sinh($a);
    $id = NumPower::subtract(NumPower::multiply($ch, $ch), NumPower::multiply($sh, $sh));
    ok(abs((float)((string)$id[0]) - 1.0) < 1e-28, "fp128 cosh²−sinh²=1 at x=$x");
}
/* Round-trips: arcsinh(sinh(x)) == x, arctanh(tanh(x)) == x. The residual
   is formed in fp128 and must be < 1e-28 (a short input string like "0.25"
   has too few digits for a direct sig() compare). */
foreach (['0.25', '1.5'] as $x) {
    $a   = new NDArray([$x], 'float128');
    $d1  = NumPower::subtract(NumPower::arcsinh(NumPower::sinh($a)), $a);
    ok(abs((float)((string)$d1[0])) < 1e-28, "fp128 arcsinh(sinh($x))==x");
    $d2  = NumPower::subtract(NumPower::arctanh(NumPower::tanh($a)), $a);
    ok(abs((float)((string)$d2[0])) < 1e-28, "fp128 arctanh(tanh($x))==x");
}

/* ── domains & special values (float64) ────────────────────────────────── */
ok(is_nan((float)NumPower::arccosh(new NDArray([0.5], 'float64'))[0]), "arccosh(0.5)=nan");
ok(is_nan((float)NumPower::arccosh(new NDArray([0.0], 'float64'))[0]), "arccosh(0)=nan");
ok((float)NumPower::arccosh(new NDArray([1.0], 'float64'))[0] === 0.0,  "arccosh(1)=0");
ok(is_nan((float)NumPower::arctanh(new NDArray([2.0], 'float64'))[0]),  "arctanh(2)=nan");
ok(is_nan((float)NumPower::arctanh(new NDArray([-2.0], 'float64'))[0]), "arctanh(-2)=nan");
$pinf = (float)NumPower::arctanh(new NDArray([1.0], 'float64'))[0];
$ninf = (float)NumPower::arctanh(new NDArray([-1.0], 'float64'))[0];
ok(is_infinite($pinf) && $pinf > 0, "arctanh(1)=+inf");
ok(is_infinite($ninf) && $ninf < 0, "arctanh(-1)=-inf");
/* tanh saturates to ±1; sinh/cosh of ±inf; nan propagation. */
ok((float)NumPower::tanh(new NDArray([50.0], 'float64'))[0] === 1.0,  "tanh(50)=1");
ok((float)NumPower::tanh(new NDArray([-50.0], 'float64'))[0] === -1.0,"tanh(-50)=-1");
ok(is_infinite((float)NumPower::sinh(new NDArray([INF], 'float64'))[0]), "sinh(inf)=inf");
ok(is_infinite((float)NumPower::cosh(new NDArray([-INF], 'float64'))[0]),"cosh(-inf)=inf");
ok((float)NumPower::tanh(new NDArray([INF], 'float64'))[0] === 1.0,   "tanh(inf)=1");
ok(is_nan((float)NumPower::sinh(new NDArray([NAN], 'float64'))[0]),   "sinh(nan)=nan");
ok(is_nan((float)NumPower::arctanh(new NDArray([NAN], 'float64'))[0]),"arctanh(nan)=nan");
/* fp128 domains too — a 0-D fp128 result renders as the strings
   "nan" / "inf" / "-inf" (PHP's (float) cast can't parse those). */
ok((string)NumPower::arccosh(new NDArray(['0.5'], 'float128'))[0] === 'nan',
   "fp128 arccosh(0.5)=nan");
ok((string)NumPower::arctanh(new NDArray(['1'], 'float128'))[0] === 'inf',
   "fp128 arctanh(1)=+inf");
ok((string)NumPower::arctanh(new NDArray(['-1'], 'float128'))[0] === '-inf',
   "fp128 arctanh(-1)=-inf");

/* ── odd/even symmetry ─────────────────────────────────────────────────── */
/* sinh/tanh/arcsinh are odd (f(−x) = −f(x)); cosh is even (f(−x) = f(x)). */
foreach ([['float64', 2.5, -2.5], ['float128', '2.5', '-2.5']] as [$dt, $x, $mx]) {
    foreach (['sinh' => true, 'tanh' => true, 'arcsinh' => true, 'cosh' => false] as $op => $odd) {
        $pos = NumPower::$op(new NDArray([$x], $dt))[0];
        $neg = NumPower::$op(new NDArray([$mx], $dt))[0];
        if ($dt === 'float64') {
            ok(near($neg, $odd ? -$pos : $pos, 1e-12), "$op symmetry $dt");
        } else {
            $ps = (string)$pos; $ns = (string)$neg;
            $mag = sig(ltrim($ps, '-'), ltrim($ns, '-')) >= 28;
            $sgn = $odd ? (($ps[0] === '-') !== ($ns[0] === '-'))
                        : (($ps[0] === '-') === ($ns[0] === '-'));
            ok($mag && $sgn, "$op symmetry $dt");
        }
    }
}

/* ── numeric-string scalar intake (float128 / uint64 inference) ─────────── */
$s = NumPower::sinh('0.5');                 /* decimal → float128 */
ok(is_string($s) && sig('0.521095305493747361622425626411491559', $s) >= 28,
   "sinh('0.5') string intake → fp128");
$u = NumPower::arcsinh('3');                /* integer → int64/uint64 → widened */
ok(near($u, asinh(3.0), 1e-12), "arcsinh('3') string intake");

/* ── shape preservation (multi-dim) ────────────────────────────────────── */
$m = NumPower::reshape(new NDArray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'float64'), [2, 3]);
$tm = NumPower::tanh($m);
ok($tm->shape() === [2, 3], "tanh keeps [2,3] shape");
ok(near($tm->toArray()[1][2], tanh(0.6), 1e-12), "tanh multidim value");
/* empty array stays empty, dtype preserved. */
$e = NumPower::sinh(new NDArray([], 'float64'));
ok($e->size() === 0 && $e->__serialize()['dtype'] === 'float64', "sinh([]) empty");

echo $FAILS === 0 ? "DONE\n" : "$FAILS FAILURE(S)\n";
?>
--EXPECT--
DONE
