--TEST--
NumPower::arctan2 — CPU↔GPU parity across every dtype, with broadcasting and multi-block sizes
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
/* arctan2 must execute entirely on the GPU for GPU-resident inputs (no CPU
   staging) and return the same values as the CPU kernels. Both paths promote
   integer / narrow-float inputs to a common float dtype first, so the result
   dtype is float-only. float128 on the GPU uses the double-double tier
   (fp64-accurate, same contract as the dd unary transcendentals), so its
   parity is checked at fp64 tolerance.

   Silent on success: the strict --EXPECT-- (single summary line) is identical
   whether or not a value diverges, so a mismatch prints a FAIL line and fails
   the test. The --SKIPIF-- gates the whole test on a present GPU. */

$FAILS = 0;
function ok($cond, $label) {
    global $FAILS;
    if (!$cond) { echo "FAIL: $label\n"; $FAILS++; }
}
function parity($x, $y, $dt, $tol) {
    $cx = NumPower::array($x, $dt);
    $cy = NumPower::array($y, $dt);
    $cpu = NumPower::arctan2($cx, $cy);
    $g   = NumPower::arctan2($cx->gpu(), $cy->gpu());
    ok($g->isGPU(), "$dt result stays on GPU");
    $gpu = $g->cpu();
    ok($cpu->__serialize()['dtype'] === $gpu->__serialize()['dtype'],
       "$dt CPU/GPU dtype match");
    $ca = $cpu->toArray(); $ga = $gpu->toArray();
    $n = count($ca); $bad = false;
    for ($i = 0; $i < $n; $i++) {
        $cv = (float)$ca[$i]; $gv = (float)$ga[$i];
        if (is_nan($cv) && is_nan($gv)) continue;
        $d = abs($cv - $gv);
        if ($d > max($tol, abs($cv) * $tol)) { $bad = true; break; }
    }
    ok(!$bad, "$dt CPU/GPU values match");
}

$x = [1.0, -1.0,  1.0, -1.0, 0.5, -0.5, 0.0,  0.0];
$y = [1.0,  1.0, -1.0, -1.0, 0.5, -0.5, 1.0, -1.0];

/* float compute dtypes */
parity($x, $y, 'float32',  1e-5);
parity($x, $y, 'float64',  1e-12);
parity($x, $y, 'float16',  2e-2);
parity(['1.0','-1.0','0.5','-0.5'], ['1.0','1.0','-0.5','0.5'], 'float128', 1e-12);

/* narrow floats — both paths route through float32 then cast back */
parity([0.0, 0.5, 1.0], [1.0, 0.5, 1.0], 'float4', 0.2);
parity([0.0, 0.5, 1.0], [1.0, 0.5, 1.0], 'float8', 0.2);

/* signed ints → float promotion */
$ix = [1, -1, 1, -1, 0]; $iy = [1, 1, -1, -1, 1];
parity($ix, $iy, 'int8',  1e-5);
parity($ix, $iy, 'int16', 1e-5);
parity($ix, $iy, 'int32', 1e-12);
parity($ix, $iy, 'int64', 1e-12);

/* unsigned ints → float promotion (non-negative inputs) */
$ux = [0, 1, 2, 3, 5]; $uy = [1, 1, 0, 4, 0];
parity($ux, $uy, 'uint8',  1e-5);
parity($ux, $uy, 'uint16', 1e-5);
parity($ux, $uy, 'uint32', 1e-12);
parity($ux, $uy, 'uint64', 1e-12);

/* ── Broadcasting parity on GPU ─────────────────────────────────────────── */
/* scalar denominator */
$bx = NumPower::array([1.0, 2.0, 3.0, 4.0], 'float64');
$cpu_bs = NumPower::arctan2($bx, 2.0)->toArray();
$gpu_bs = NumPower::arctan2($bx->gpu(), 2.0);
ok($gpu_bs->isGPU(), 'broadcast scalar stays on GPU');
$gpu_bs = $gpu_bs->cpu()->toArray();
$bs_ok = true;
for ($i = 0; $i < 4; $i++) if (abs($cpu_bs[$i] - $gpu_bs[$i]) > 1e-12) $bs_ok = false;
ok($bs_ok, 'broadcast scalar CPU/GPU parity');

/* row vector → matrix */
$m = NumPower::array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], 'float64');
$r = NumPower::array([1.0, 2.0, 3.0], 'float64');
$cpu_m = NumPower::arctan2($m, $r)->toArray();
$gpu_m = NumPower::arctan2($m->gpu(), $r->gpu())->cpu()->toArray();
$m_ok = true;
for ($i = 0; $i < 2; $i++) for ($j = 0; $j < 3; $j++)
    if (abs($cpu_m[$i][$j] - $gpu_m[$i][$j]) > 1e-12) $m_ok = false;
ok($m_ok, 'broadcast row-vector → matrix CPU/GPU parity');

/* ── Mixed device: a CPU scalar operand migrates to the GPU array's device ─ */
$ga = NumPower::array([1.0, -1.0, 2.0, -2.0], 'float64')->gpu();
$mixed = NumPower::arctan2(2.0, $ga);          /* CPU scalar numerator + GPU array */
ok($mixed->isGPU(), 'mixed CPU-scalar + GPU-array stays on GPU');
$ref = NumPower::arctan2(2.0, NumPower::array([1.0, -1.0, 2.0, -2.0], 'float64'))->toArray();
$mx  = $mixed->cpu()->toArray();
$mok = true;
for ($i = 0; $i < 4; $i++) if (abs($ref[$i] - $mx[$i]) > 1e-12) $mok = false;
ok($mok, 'mixed CPU-scalar + GPU-array parity');

/* ── Multi-block size (N = 4097 spans > 1 CUDA block of 256) ─────────────── */
$N = 4097; $lx = []; $ly = [];
for ($i = 0; $i < $N; $i++) { $lx[$i] = (($i % 11) - 5) * 0.3; $ly[$i] = (($i % 7) - 3) * 0.4; }
foreach (['float32' => 1e-5, 'float64' => 1e-12] as $dt => $tol) {
    $cx = NumPower::array($lx, $dt); $cy = NumPower::array($ly, $dt);
    $cpu = NumPower::arctan2($cx, $cy)->toArray();
    $gpu = NumPower::arctan2($cx->gpu(), $cy->gpu())->cpu()->toArray();
    $big_ok = true;
    for ($i = 0; $i < $N; $i++) {
        $cv = (float)$cpu[$i]; $gv = (float)$gpu[$i];
        if (is_nan($cv) && is_nan($gv)) continue;
        if (abs($cv - $gv) > max($tol, abs($cv) * $tol)) { $big_ok = false; break; }
    }
    ok($big_ok, "$dt multi-block (N=$N) CPU/GPU parity");
}

echo $FAILS === 0 ? "ALL CHECKS PASSED\n" : "TOTAL FAILURES: $FAILS\n";
?>
--EXPECT--
ALL CHECKS PASSED
