--TEST--
GPU DD-precision regressions (dd_two_prod, exp/log1p/sinc fp128, logb), NaN-norm, clip saturation, CPU/GPU parity
--SKIPIF--
<?php
try {
    $a = NumPower::array([1.0])->gpu();
    if (!$a->isGPU()) die("skip GPU not available");
} catch (Throwable $t) {
    die("skip GPU not available: " . $t->getMessage());
}
?>
--FILE--
<?php
/* Guards the GPU-side precision fixes on this branch:

   - dd_two_prod: the `__dmul_rn` fix that stops NVCC's -fmad contraction
     from collapsing DD multiply to fp64. Exercised through GPU fp128
     multiply, which must agree with CPU libquadmath to full DD precision.
   - exp (24-term Taylor) / log1p (atanh substitution for small |x|) /
     sinc (12-term series over |x|<=0.1): each previously lost precision
     on GPU (exp ~29 digits, log1p ~16, sinc ~15 at the boundary).
   - logb: integer-valued, must be exact and CPU==GPU.
   - NaN-sign normalization holds on GPU-resident arrays.
   - integer clip-bound saturation holds on GPU.
   - float32/float64 transcendentals stay CPU==GPU within fp tolerance.

   fp128 agreement is measured NUMERICALLY (not by character match), so
   it is robust to CPU libquadmath printing a short rounded form
   (e.g. log1p(1e-20) → "...9995e-21") while the GPU prints the full DD
   form ("...94999...e-21"): both are the same number to ~32 digits. */

/* Parse a decimal string ("9.99e-21") into [neg, intDigits, exp10] with
   value = (neg?-1:1) * intDigits * 10^exp10. */
function ndps_parse($s) {
    $s = trim($s); $neg = false;
    if ($s !== '' && $s[0] === '-') { $neg = true; $s = substr($s, 1); }
    elseif ($s !== '' && $s[0] === '+') { $s = substr($s, 1); }
    $exp = 0;
    if (($p = stripos($s, 'e')) !== false) { $exp = (int)substr($s, $p + 1); $s = substr($s, 0, $p); }
    if (($d = strpos($s, '.')) !== false) { $exp -= (strlen($s) - $d - 1); $s = substr($s, 0, $d) . substr($s, $d + 1); }
    $s = ltrim($s, '0'); if ($s === '') $s = '0';
    return [$neg, $s, $exp];
}
function ndps_cmp($A, $B) {
    if (strlen($A) != strlen($B)) return strlen($A) <=> strlen($B);
    return strcmp($A, $B);
}
function ndps_sub($A, $B) { /* |A|-|B|, assumes A>=B, digit strings */
    $A = strrev($A); $B = strrev($B); $r = ''; $carry = 0;
    for ($i = 0; $i < strlen($A); $i++) {
        $da = ord($A[$i]) - 48; $db = $i < strlen($B) ? ord($B[$i]) - 48 : 0;
        $d = $da - $db - $carry; if ($d < 0) { $d += 10; $carry = 1; } else $carry = 0;
        $r .= chr($d + 48);
    }
    $r = ltrim(strrev($r), '0'); return $r === '' ? '0' : $r;
}
/* Significant-digit agreement between two non-negative decimal strings. */
function agree_digits($a, $b) {
    [$na, $da, $ea] = ndps_parse($a); [$nb, $db, $eb] = ndps_parse($b);
    if ($da === '0' && $db === '0') return 99;
    $common = min($ea, $eb);
    $A = $da . str_repeat('0', $ea - $common);
    $B = $db . str_repeat('0', $eb - $common);
    $diff = (ndps_cmp($A, $B) >= 0) ? ndps_sub($A, $B) : ndps_sub($B, $A);
    if ($diff === '0') return 99;
    $loga = $ea + strlen($da) - 1;
    $logd = $common + strlen($diff) - 1;
    return $loga - $logd;
}

$T = 30;
function fp128_parity($op, $v, $T) {
    $c = new NDArray([$v], 'float128');
    $rc = (string)NumPower::$op($c)->toArray()[0];
    $rg = (string)NumPower::$op($c->gpu())->cpu()->toArray()[0];
    $m = agree_digits($rc, $rg);
    echo ($m >= $T) ? "OK $op($v): >= $T digits (CPU==GPU)\n"
                    : "FAIL $op($v): only $m digits\n  CPU=$rc\n  GPU=$rg\n";
}

/* ── dd_two_prod regression: GPU fp128 multiply ──────────────────────── */
echo "=== dd_two_prod (GPU fp128 multiply) ===\n";
foreach ([
    ["1.4142135623730950488016887242097", "1.7320508075688772935274463415059"],
    ["3.1415926535897932384626433832795", "2.7182818284590452353602874713527"],
    ["1.0000000000000001", "1.0000000000000001"],
] as [$x, $y]) {
    $a = new NDArray([$x], 'float128');
    $b = new NDArray([$y], 'float128');
    $rc = (string)NumPower::multiply($a, $b)->toArray()[0];
    $rg = (string)NumPower::multiply($a->gpu(), $b->gpu())->cpu()->toArray()[0];
    $m = agree_digits($rc, $rg);
    echo ($m >= $T) ? "OK fp128 multiply: >= $T digits (CPU==GPU)\n"
                    : "FAIL fp128 multiply $x*$y: only $m digits\n  CPU=$rc\n  GPU=$rg\n";
}

/* ── exp / log1p / sinc fp128 GPU precision ──────────────────────────── */
echo "\n=== exp / log1p / sinc fp128 GPU precision ===\n";
fp128_parity('exp',   '0.3465735902799726', $T);   // worst-case reduced arg
fp128_parity('exp',   '1.0',                $T);
fp128_parity('exp2',  '0.1',                $T);
fp128_parity('expm1', '1.0',                $T);
fp128_parity('log1p', '1e-20', $T);                 // sub-fp64-eps: was ~16 digits
fp128_parity('log1p', '1e-30', $T);
fp128_parity('log1p', '0.4',   $T);
fp128_parity('sinc',  '0.1',   $T);                 // Taylor upper boundary
fp128_parity('sinc',  '0.05',  $T);
fp128_parity('log',   '1.5',   $T);
fp128_parity('log10', '7.0',   $T);

/* ── logb fp128 exact + CPU==GPU ─────────────────────────────────────── */
echo "\n=== logb fp128 (exact, CPU==GPU) ===\n";
foreach (['12.5' => '3', '1024.0' => '10', '0.1' => '-4', '1.0' => '0'] as $v => $want) {
    $rg = (string)NumPower::logb((new NDArray([$v], 'float128'))->gpu())->cpu()->toArray()[0];
    echo ($rg === $want) ? "OK logb($v) = $want\n" : "FAIL logb($v): got $rg want $want\n";
}

/* ── NaN-sign normalization on GPU ───────────────────────────────────── */
echo "\n=== NaN normalization (GPU) ===\n";
foreach (['float32', 'float64', 'float128'] as $dt) {
    $g = NumPower::log((new NDArray([-1.0], $dt))->gpu());
    $s = (string)$g;
    $ok = (strpos($s, 'nan') !== false) && (strpos($s, '-nan') === false)
        && (strpos($s, 'NAN') === false);
    echo "$dt: " . ($ok ? "OK" : "FAIL ($s)") . "\n";
}

/* ── integer clip saturation on GPU ──────────────────────────────────── */
echo "\n=== integer clip saturation (GPU) ===\n";
function gclip($label, $dt, $data, $lo, $hi) {
    $r = NumPower::clip((new NDArray($data, $dt))->gpu(), $lo, $hi)->cpu()->toArray();
    echo "$label: [" . implode(",", $r) . "]\n";
}
gclip("int8  clip(-50,50)",  "int8",  [-100,0,100,127,-128], "-50", "50");
gclip("int8  clip(-inf,10)", "int8",  [-50,0,50],            "-inf", "10");
gclip("uint8 clip(-50,150)", "uint8", [0,100,200,255],       "-50", "150");
gclip("int32 clip(-1e9,1e9)","int32", [-2000000000,0,2000000000], "-1000000000", "1000000000");

/* ── float32 / float64 CPU==GPU parity ───────────────────────────────── */
echo "\n=== float32/float64 transcendental CPU==GPU parity ===\n";
$ops = ['exp','exp2','expm1','log','log1p','log2','log10','logb'];
foreach (['float32','float64'] as $dt) {
    $vals = [0.5,1.0,2.0,8.0,100.0];
    foreach ($ops as $op) {
        $cpu = NumPower::$op(NumPower::array($vals,$dt))->toArray();
        $gpu = NumPower::$op(NumPower::array($vals,$dt)->gpu())->cpu()->toArray();
        $maxrel = 0;
        for ($i=0;$i<count($vals);$i++){ $r=abs($cpu[$i]-$gpu[$i])/(abs($cpu[$i])+1e-30); if($r>$maxrel)$maxrel=$r; }
        $tol = ($dt==='float32') ? 1e-5 : 1e-12;
        echo "$dt $op: " . ($maxrel <= $tol ? "OK" : sprintf("FAIL maxrel=%.2e",$maxrel)) . "\n";
    }
}

echo "DONE\n";
?>
--EXPECT--
=== dd_two_prod (GPU fp128 multiply) ===
OK fp128 multiply: >= 30 digits (CPU==GPU)
OK fp128 multiply: >= 30 digits (CPU==GPU)
OK fp128 multiply: >= 30 digits (CPU==GPU)

=== exp / log1p / sinc fp128 GPU precision ===
OK exp(0.3465735902799726): >= 30 digits (CPU==GPU)
OK exp(1.0): >= 30 digits (CPU==GPU)
OK exp2(0.1): >= 30 digits (CPU==GPU)
OK expm1(1.0): >= 30 digits (CPU==GPU)
OK log1p(1e-20): >= 30 digits (CPU==GPU)
OK log1p(1e-30): >= 30 digits (CPU==GPU)
OK log1p(0.4): >= 30 digits (CPU==GPU)
OK sinc(0.1): >= 30 digits (CPU==GPU)
OK sinc(0.05): >= 30 digits (CPU==GPU)
OK log(1.5): >= 30 digits (CPU==GPU)
OK log10(7.0): >= 30 digits (CPU==GPU)

=== logb fp128 (exact, CPU==GPU) ===
OK logb(12.5) = 3
OK logb(1024.0) = 10
OK logb(0.1) = -4
OK logb(1.0) = 0

=== NaN normalization (GPU) ===
float32: OK
float64: OK
float128: OK

=== integer clip saturation (GPU) ===
int8  clip(-50,50): [-50,0,50,50,-50]
int8  clip(-inf,10): [-50,0,10]
uint8 clip(-50,150): [0,100,150,150]
int32 clip(-1e9,1e9): [-1000000000,0,1000000000]

=== float32/float64 transcendental CPU==GPU parity ===
float32 exp: OK
float32 exp2: OK
float32 expm1: OK
float32 log: OK
float32 log1p: OK
float32 log2: OK
float32 log10: OK
float32 logb: OK
float64 exp: OK
float64 exp2: OK
float64 expm1: OK
float64 log: OK
float64 log1p: OK
float64 log2: OK
float64 log10: OK
float64 logb: OK
DONE
